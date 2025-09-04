# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np

from qlib.data.dataset.processor import Processor
from qlib.log import get_module_logger

logger = get_module_logger("BarraProcessor")

def normalize_ticker(x: str) -> str:
    """Normalize raw ticker like '300058.SZ' -> 'SZ300058' for Qlib.

    If exchange suffix is present (SZ/SH/BJ), move it to prefix and zero-pad code to 6 digits.
    Otherwise, return the upper-cased stripped string.
    """
    s = str(x).strip().upper()
    if "." in s:
        base, exch = s.split(".")
        base = base[-6:].zfill(6)
        if exch in ("SZ", "SH", "BJ"):
            return f"{exch}{base}"
    return s


def trim_outliers(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    """Mask values outside [lower_q, upper_q] quantiles with NaN.

    Expects a numeric Series. Inf values are treated as NaN for quantile calculation.
    If quantiles are invalid or identical, returns the original series.
    """
    s = pd.to_numeric(series, errors="coerce")
    finite = s.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return s
    ql = finite.quantile(lower_q)
    qh = finite.quantile(upper_q)
    if pd.isna(ql) or pd.isna(qh) or not (qh > ql):
        return s
    return s.mask((s < ql) | (s > qh))


class BarraCNE6Processor(Processor):
    """Processor to compute and attach Barra-style exposures via CNE6 functions.

    It expects the input DataFrame to be MultiIndex (datetime, instrument),
    with columns including at least:
      - "$return" (or precomputed returns)
      - "$market_cap"
      - a market returns column (default "$market_ret")
    Optionally:
      - turnover column
      - EPS trailing and predicted columns
      - price column (default "$close")
    The processor will compute exposures on the full period and attach columns
    "B_SIZE", "B_BETA", "B_MOM", "B_RESVOL", and when data available:
    "B_VOL", "B_LIQ", "B_EY".
    """

    def __init__(
        self,
        *,
        market_col: str = "$market_ret",
        turnover_col: str = "$turnover",
        eps_trailing_col: Optional[str] = None,
        eps_pred_col: Optional[str] = None,
        price_col: str = "$close",
        lookbacks: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.market_col = market_col
        self.turnover_col = turnover_col
        self.eps_trailing_col = eps_trailing_col
        self.eps_pred_col = eps_pred_col
        self.price_col = price_col
        self.lookbacks = lookbacks or {}

    def fit(self, df: pd.DataFrame):
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        assert isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2, "Input df must be MultiIndex (datetime, instrument)"

        def pivot(col: str) -> Optional[pd.DataFrame]:
            if col is None or col not in df.columns:
                return None
            wide = df[col].unstack(level="instrument").sort_index()
            return wide

        # Required
        returns = pivot("$return")
        if returns is None:
            # Try derive simple returns from close
            close = pivot(self.price_col)
            if close is not None:
                returns = close.pct_change()
        mcap = pivot("$market_cap")
        market = pivot(self.market_col)
        market_series = market.iloc[:, 0] if market is not None else None

        # Optional
        turnover = pivot(self.turnover_col) if self.turnover_col else None
        eps_trailing = pivot(self.eps_trailing_col) if self.eps_trailing_col else None
        eps_pred = pivot(self.eps_pred_col) if self.eps_pred_col else None
        price = pivot(self.price_col) if self.price_col else None

        if returns is None or mcap is None or market_series is None:
            return df

        expos = compute_cne6_exposures(
            returns=returns,
            mcap=mcap,
            market_returns=market_series,
            turnover=turnover,
            eps_trailing=eps_trailing,
            eps_pred=eps_pred,
            price=price,
            lookbacks=self.lookbacks,
        )
        if expos is None or expos.empty:
            return df

        # Attach back to long format
        # Map exposures (indexed by instrument) to every row by instrument level
        if not expos.empty:
            inst_index = df.index.get_level_values("instrument")
            assign = {}
            for col in expos.columns:
                mapping = expos[col]
                assign[f"B_{col}"] = inst_index.map(mapping).astype(float)
            if assign:
                df = df.assign(**assign)
        return df


class FundamentalProfileProcessor(Processor):
    """Load cn_profile fundamentals and align by announcement windows.

    - Source: data/others/cn_profile.csv
    - Columns required: ticker, period, ann_date, <factor columns>
    - Values are valid from current ann_date until next ann_date for the same ticker.
    """

    def __init__(self, csv_path: str = "data/others/cn_profile.csv", prefix: str = "F_", date_col: str = "ann_date"):
        super().__init__()
        self.csv_path = csv_path
        self.prefix = prefix
        self.date_col = date_col
        self._cache = None

    def fit(self, df: pd.DataFrame):
        return self

    def _load_profile(self) -> pd.DataFrame:
        prof = pd.read_csv(self.csv_path)
        # Normalize columns
        prof.columns = [str(c).strip().lower() for c in prof.columns]
        assert "ticker" in prof.columns and self.date_col in prof.columns, "CSV must include ticker and ann_date"

        # Robust ann_date parsing: accept 'YYYY-MM-DD' or numeric 'YYYYMMDD'
        raw_ad = prof[self.date_col]
        # Filter out null ann_date before parsing
        raw_ad = raw_ad.dropna()

        # Handle possible float by rounding to int and str
        try:
            raw_vals = pd.to_numeric(raw_ad, errors='coerce')
            # Coerce to int, format as 8-digit string
            raw_ad = raw_vals.round().astype('Int64').apply(lambda x: f"{x:08d}" if pd.notna(x) else "")
            ad = pd.to_datetime(raw_ad, format='%Y%m%d', errors='coerce')
        except Exception:
            raw_ad = raw_ad.astype(str).str.strip()
            ad = pd.to_datetime(raw_ad, errors='coerce')
        prof[self.date_col] = ad.dt.normalize()  # Ensure date-only, no time component

        # Normalize ticker to Qlib format, e.g., 300058.SZ -> SZ300058
        prof["ticker"] = prof["ticker"].map(normalize_ticker)

        # Coerce potential numeric fields to numeric
        key_cols = {"ticker", self.date_col, "period"}
        factor_cols_all = [c for c in prof.columns if c not in key_cols]
        ratio_like = {
            "gross_margin",
            "operating_margin",
            "net_margin",
            "return_on_equity",
            "return_on_assets",
            "debt_to_equity",
            "debt_to_assets",
            "return_on_invested_capital",
            "revenue_growth",
            "total_assets_growth",
        }
        for c in factor_cols_all:
            series = prof[c]
            if series.dtype == object:
                cleaned = (
                    series.astype(str)
                    .str.strip()
                    .str.replace("%", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace(r"[\u2212\u2013\u2014]", "-", regex=True)
                )
                col_series = pd.to_numeric(cleaned, errors="coerce")
            else:
                col_series = pd.to_numeric(series, errors="coerce")
            # Scale percentage only for ratio-like columns
            if c in ratio_like:
                finite = col_series.replace([np.inf, -np.inf], np.nan).dropna()
                if not finite.empty and finite.quantile(0.95) > 1.5:
                    col_series = col_series / 100.0

            # Trim outliers: keep within [0.05, 0.95] quantile range
            #col_series = trim_outliers(col_series, 0.05, 0.95)
            prof[c] = col_series

        prof = prof.dropna(subset=[self.date_col])  # Drop invalid ann_date
        prof.sort_values(["ticker", self.date_col], inplace=True)
        return prof

    def _extract_index_levels(self, df: pd.DataFrame) -> Tuple[int, int]:
        idx_names = list(df.index.names)
        inst_level = idx_names.index("instrument") if "instrument" in idx_names else 1
        date_level = idx_names.index("datetime") if "datetime" in idx_names else 0
        return inst_level, date_level

    def _prepare_trade_data(self, df: pd.DataFrame, inst_level: int, date_level: int) -> Tuple[pd.DatetimeIndex, set]:
        trade_dates = pd.to_datetime(df.index.get_level_values(date_level).unique()).normalize().sort_values()
        codes = df.index.get_level_values(inst_level).unique()
        code_set = set(map(str, codes))
        return trade_dates, code_set

    def _build_merged_fundamentals(self, prof: pd.DataFrame, trade_dates: pd.DatetimeIndex, code_set: set) -> pd.DataFrame:
        key_cols = {"ticker", self.date_col, "period"}
        factor_cols = [c for c in prof.columns if c not in key_cols]
        if not factor_cols:
            return pd.DataFrame()

        frames = []
        for ticker, g in prof.groupby("ticker"):
            if str(ticker) not in code_set:
                continue
            ts = g.set_index(self.date_col)[factor_cols].sort_index()
            ts.index = pd.to_datetime(ts.index).normalize()  # Normalize ann_date to date-only Timestamp
            if len(trade_dates) == 0:
                continue
            idx_pos = trade_dates.searchsorted(ts.index, side="right")
            idx_pos = np.clip(idx_pos, 0, len(trade_dates) - 1)
            eff_index = trade_dates[idx_pos]
            ts.index = eff_index
            # Drop duplicate effective dates per ticker, keep the latest record
            if ts.index.has_duplicates:
                ts = ts[~ts.index.duplicated(keep="last")]
            ts = ts.sort_index()  # Ensure monotonic after drop
            # Reindex to all trade dates and forward-fill
            aligned = ts.reindex(trade_dates).ffill()
            aligned["instrument"] = ticker
            aligned["datetime"] = aligned.index
            frames.append(aligned)
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        merged.set_index(["datetime", "instrument"], inplace=True)
        merged.sort_index(inplace=True)
        return merged

    def _reorder_merged_index(self, merged: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        if list(merged.index.names) != list(df.index.names):
            merged = merged.reorder_levels(df.index.names).sort_index()
        return merged

    def _assign_factors(self, merged: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        assign_map = {}
        factor_cols = [c for c in merged.columns]
        for col in factor_cols:
            out = f"{self.prefix}{col.upper()}"
            ser = merged[col].groupby(level=1).ffill()
            assign_map[out] = ser.reindex(df.index).astype(float)
        if assign_map:
            df = df.assign(**assign_map)
        return df

    def _derive_core_fields(self, merged: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        get = lambda c: merged.get(c, pd.Series(index=merged.index, dtype=float))
        me = get("market_cap")
        ld = get("long_term_debt")
        td = get("total_debt")
        st_debt = get("short_term_debt")
        if td.isna().all():
            td = ld.fillna(0.0) + st_debt.fillna(0.0)
        be = get("shareholders_equity")
        tl = get("total_liabilities")
        ta = get("total_assets")
        sales = get("revenue")
        gross = get("gross_profit")
        cogs = sales - gross
        earnings = get("net_income")
        cash = get("cash_and_equivalents")
        cfo = get("operating_cash_flow")
        capex = get("capital_expenditure")
        cfi = -capex
        da = get("depreciation_and_amortization")

        # Improve total debt fallback
        if td.isna().all():
            td_fallback = ld.fillna(0.0) + st_debt.fillna(0.0)
            both_nan = ld.isna() & st_debt.isna()
            td = td_fallback.mask(both_nan, np.nan)

        derived = {
            "F_LD": ld,
            "F_TD": td,
            "F_BE": be,
            "F_TL": tl,
            "F_TA": ta,
            "F_SALES": sales,
            "F_COGS": cogs,
            "F_EARNINGS": earnings,
            "F_CASH": cash,
            "F_CFO": cfo,
            "F_CFI": cfi,
            "F_DA": da,
        }
        assign_map = {}
        for out, ser in derived.items():
            if ser.empty:
                continue
            ser = ser.groupby(level=1).ffill()
            assign_map[out] = ser.reindex(df.index).astype(float)
        if assign_map:
            df = df.assign(**assign_map)
        return df

    def _set_market_equity(self, merged: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        get = lambda c: merged.get(c, pd.Series(index=merged.index, dtype=float))
        me = get("market_cap")
        if "$market_cap" in df.columns:
            df["F_ME"] = df["$market_cap"].astype(float)
        elif not me.empty:
            me = me.groupby(level=1).ffill()
            df["F_ME"] = me.reindex(df.index).astype(float)
        return df

    def _assign_ratios(self, merged: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        ratio_map = {
            "F_DTOA_RATIO": "debt_to_assets",
            "F_DEBT_TO_EQUITY": "debt_to_equity",
            "F_GROSS_MARGIN": "gross_margin",
            "F_OPERATING_MARGIN": "operating_margin",
            "F_NET_MARGIN": "net_margin",
            "F_ROA_RATIO": "return_on_assets",
            "F_ROE_RATIO": "return_on_equity",
            "F_ROIC": "return_on_invested_capital",
            "F_REVENUE_GROWTH": "revenue_growth",
            "F_TOTAL_ASSETS_GROWTH": "total_assets_growth",
        }
        ratio_assign = {}
        for out, colname in ratio_map.items():
            if colname in merged.columns:
                ser = merged[colname].groupby(level=1).ffill().reindex(df.index).astype(float)
                ratio_assign[out] = ser
        if ratio_assign:
            df = df.assign(**ratio_assign)

        # Fallback: if F_ROE_RATIO still missing/NaN but F_RETURN_ON_EQUITY exists from factor assignment, copy it
        if "F_RETURN_ON_EQUITY" in df.columns:
            if ("F_ROE_RATIO" not in df.columns) or df["F_ROE_RATIO"].isna().all():
                df["F_ROE_RATIO"] = df["F_RETURN_ON_EQUITY"].astype(float)
            else:
                df["F_ROE_RATIO"] = df["F_ROE_RATIO"].where(~df["F_ROE_RATIO"].isna(), df["F_RETURN_ON_EQUITY"].astype(float))
        return df

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        assert isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2, "Input df must be MultiIndex (datetime, instrument)"

        if self._cache is None:
            self._cache = self._load_profile()
        prof = self._cache

        inst_level, date_level = self._extract_index_levels(df)
        trade_dates, code_set = self._prepare_trade_data(df, inst_level, date_level)

        merged = self._build_merged_fundamentals(prof, trade_dates, code_set)
        if merged.empty:
            return df
        merged = self._reorder_merged_index(merged, df)
        df = self._assign_factors(merged, df)
        df = self._derive_core_fields(merged, df)
        df = self._set_market_equity(merged, df)
        df = self._assign_ratios(merged, df)
        # Attach industry info if available via IndustryProcessor
        try:
            df = IndustryProcessor()(df)
        except Exception:
            # Silently skip if industry mapping not available
            pass
        return df


class BarraFundamentalQualityProcessor(Processor):
    """Compute Barra CNE6 quality-related factors from fundamental columns.

    This processor expects the DataFrame to already contain fundamental columns from
    FundamentalProfileProcessor (prefixed by F_), such as:
      - F_ME, F_PE, F_LD, F_BE, F_TL, F_TA, F_SALES, F_COGS, F_EARNINGS,
        F_CASH, F_TD, F_CFO, F_CFI, F_DA

    It will compute and attach the following quality factors:
      - Q_MLEV, Q_BLEV, Q_DTOA, Q_ATO, Q_GP, Q_GPM, Q_ROA, Q_ABS, Q_ACF
    """

    def fit(self, df: pd.DataFrame):
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        assert isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2

        def col(name: str) -> pd.Series:
            return df.get(name, pd.Series(index=df.index, dtype=float))

        def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            val = a / b.replace(0, np.nan)
            return val.replace([np.inf, -np.inf], np.nan)

        # Basic components
        ME = col("F_ME").astype(float)
        PE = col("F_PE").astype(float).fillna(0.0)
        LD = col("F_LD").astype(float).fillna(0.0)
        BE = col("F_BE").astype(float)
        TL = col("F_TL").astype(float)
        TA = col("F_TA").astype(float)
        SALES = col("F_SALES").astype(float)
        COGS = col("F_COGS").astype(float)
        EARN = col("F_EARNINGS").astype(float)
        CASH = col("F_CASH").astype(float).fillna(0.0)
        TD = col("F_TD").astype(float).fillna(0.0)
        CFO = col("F_CFO").astype(float).fillna(0.0)
        CFI = col("F_CFI").astype(float).fillna(0.0)
        DA = col("F_DA").astype(float).fillna(0.0)

        # Prefer ratio columns if available
        q_assign = {}
        if "F_DTOA_RATIO" in df.columns:
            q_assign["Q_DTOA"] = df["F_DTOA_RATIO"]
        if "F_GROSS_MARGIN" in df.columns:
            q_assign["Q_GPM"] = df["F_GROSS_MARGIN"]
        if "F_OPERATING_MARGIN" in df.columns:
            q_assign["Q_OPM"] = df["F_OPERATING_MARGIN"]
        if "F_NET_MARGIN" in df.columns:
            q_assign["Q_NPM"] = df["F_NET_MARGIN"]
        if "F_ROA_RATIO" in df.columns:
            q_assign["Q_ROA"] = df["F_ROA_RATIO"]
        # Attach total assets growth only when present
        if "Q_TAGR" not in q_assign and "F_TOTAL_ASSETS_GROWTH" in df.columns:
            q_assign["Q_TAGR"] = df["F_TOTAL_ASSETS_GROWTH"]

        # Leverage (fallback)
        if "Q_MLEV" not in q_assign:
            q_assign["Q_MLEV"] = safe_div(ME + PE + LD, ME)
        if "Q_BLEV" not in q_assign:
            q_assign["Q_BLEV"] = safe_div(BE + PE + LD, ME)
        if "Q_DTOA" not in q_assign:
            q_assign["Q_DTOA"] = safe_div(TL, TA)

        # Profitability & turnover
        GP = SALES - COGS
        if "Q_ATO" not in q_assign:
            q_assign["Q_ATO"] = safe_div(SALES, TA)
        if "Q_GP" not in q_assign:
            q_assign["Q_GP"] = safe_div(GP, TA)
        if "Q_GPM" not in q_assign:
            q_assign["Q_GPM"] = safe_div(GP, SALES)
        if "Q_ROA" not in q_assign:
            q_assign["Q_ROA"] = safe_div(EARN, TA)

        # Accruals
        NOA = (TA - CASH) - (TL - TD)
        dNOA = NOA.groupby(level=1).diff()
        ACCR_BS = dNOA - DA
        if "Q_ABS" not in q_assign:
            q_assign["Q_ABS"] = -safe_div(ACCR_BS, TA)

        ACCR_CF = EARN - (CFO + CFI) + DA
        if "Q_ACF" not in q_assign:
            q_assign["Q_ACF"] = -safe_div(ACCR_CF, TA)

        if q_assign:
            df = df.assign(**q_assign)

        return df


class IndustryProcessor(Processor):
    """Attach industry code/name per ticker from a company facts file.

    - Source: data/others/cn_company_facts.csv (configurable)
    - Required columns: ticker, industry_code, industry
    - Values are static per ticker and will be attached to every date row.
    """

    def __init__(self, csv_path: str = "data/others/cn_company_facts.csv", prefix: str = "F_", code_col: str = "industry_code", name_col: str = "industry"):
        super().__init__()
        self.csv_path = csv_path
        self.prefix = prefix
        self.code_col = code_col
        self.name_col = name_col
        self._cache = None  # DataFrame with columns: ticker, code_col, name_col

    def fit(self, df: pd.DataFrame):
        return self

    def _load_mapping(self) -> pd.DataFrame:
        # Try reading as CSV; if missing extension, try appending .csv
        try:
            facts = pd.read_csv(self.csv_path, low_memory=False)
        except Exception:
            try:
                facts = pd.read_csv(f"{self.csv_path}.csv", low_memory=False)
            except Exception as e2:
                raise FileNotFoundError(f"Failed to read company facts from {self.csv_path}: {e2}")

        facts.columns = [str(c).strip().lower() for c in facts.columns]
        if "ticker" not in facts.columns:
            raise ValueError("cn_company_facts must include a 'ticker' column")
        if self.code_col not in facts.columns and self.name_col not in facts.columns:
            raise ValueError(f"cn_company_facts must include '{self.code_col}' or '{self.name_col}' column")

        # Normalize tickers to Qlib format, e.g., 300058.SZ -> SZ300058
        facts["ticker"] = facts["ticker"].map(normalize_ticker)

        # Keep only necessary columns
        keep_cols = ["ticker"] + [c for c in [self.code_col, self.name_col] if c in facts.columns]
        facts = facts[keep_cols].copy()

        # Deduplicate by ticker: keep last non-null entries
        facts.sort_values(["ticker"], inplace=True)
        facts = facts.drop_duplicates(subset=["ticker"], keep="last")
        return facts

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        assert isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2, "Input df must be MultiIndex (datetime, instrument)"

        if self._cache is None:
            self._cache = self._load_mapping()
        facts = self._cache
        if facts.empty:
            return df

        # Build mapping dicts
        code_map = facts.set_index("ticker")[self.code_col].to_dict() if self.code_col in facts.columns else {}
        name_map = facts.set_index("ticker")[self.name_col].to_dict() if self.name_col in facts.columns else {}

        inst_index = df.index.get_level_values("instrument").astype(str)
        assign = {}
        if code_map:
            assign[f"{self.prefix}INDUSTRY_CODE"] = inst_index.map(code_map)
        if name_map:
            assign[f"{self.prefix}INDUSTRY"] = inst_index.map(name_map)
        if assign:
            df = df.assign(**assign)
        return df


class IndustryMomentumProcessor(Processor):
    """Compute Industry Momentum per instrument using vectorized convolution.

    Requirements:
      - '$return' or '$close' to compute returns
      - '$market_cap' for weights
      - 'F_INDUSTRY_CODE' attached by IndustryProcessor
    Output column: 'B_INDMOM'
    """

    def __init__(
        self,
        *,
        return_col: str = "$return",
        close_col: str = "$close",
        mcap_col: str = "$market_cap",
        industry_col: str = "F_INDUSTRY_CODE",
        window: int = 126,
        halflife: int = 21,
        out_col: str = "B_INDMOM",
        debug: bool = False,
    ):
        super().__init__()
        self.return_col = return_col
        self.close_col = close_col
        self.mcap_col = mcap_col
        self.industry_col = industry_col
        self.window = int(window)
        self.halflife = int(halflife)
        self.out_col = out_col
        self.debug = debug

    def fit(self, df: pd.DataFrame):
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        assert isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2, "Input df must be MultiIndex (datetime, instrument)"

        def pivot(col: str) -> Optional[pd.DataFrame]:
            if col not in df.columns:
                return None
            return df[col].unstack(level="instrument").sort_index().astype(np.float32)

        rets = pivot(self.return_col)
        if rets is None:
            close = pivot(self.close_col)
            if close is None:
                return df
            rets = close.pct_change()
        mcap = pivot(self.mcap_col)
        if mcap is None:
            return df

        if self.industry_col not in df.columns:
            return df
        ind_map = df[self.industry_col].groupby(level="instrument").last()
        ind_map = ind_map.dropna().astype(str)
        if ind_map.empty:
            return df

        # Ensure index/column names for stable stacking/join
        rets.index.name = "datetime"
        rets.columns.name = "instrument"
        if mcap is not None:
            mcap.index.name = "datetime"
            mcap.columns.name = "instrument"

        common_cols = ind_map.index.intersection(rets.columns)
        if len(common_cols) == 0:
            return df
        rets = rets[common_cols]
        mcap = mcap.reindex(columns=common_cols)

        # Weighted rolling sum of log(1+r) using vectorized convolution per column
        weights = np.exp(-np.log(2) * np.arange(self.window) / max(self.halflife, 1))[::-1]
        weights /= weights.sum()
        log1p = np.log1p(rets.fillna(0.0)).to_numpy()
        rs_np = np.full(log1p.shape, np.nan, dtype=np.float32)
        w_flip = weights[::-1].astype(np.float32)
        for j in range(log1p.shape[1]):
            rs_col = convolve(log1p[:, j], w_flip, mode='valid')
            rs_np[self.window - 1 :, j] = rs_col.astype(np.float32)
        rs_df = pd.DataFrame(rs_np, index=rets.index, columns=rets.columns)
        try:
            logger.info(f"INDMOM rs_df non-null={int(rs_df.notna().sum().sum())}")
        except Exception:
            pass

        # Weights proportional to sqrt(market cap); ffill to avoid NaNs on missing days
        mcap = mcap.replace([np.inf, -np.inf], np.nan).ffill()
        w_raw = np.sqrt(mcap.clip(lower=0)).replace([np.inf, -np.inf], np.nan)
        w_sum = w_raw.sum(axis=1).replace(0, np.nan)
        w_norm = w_raw.div(w_sum, axis=0)

        # Group instruments by industry
        ind_to_cols = {ind: list(g.index) for ind, g in ind_map.groupby(ind_map)}
        industry_rs_df = pd.DataFrame(index=rs_df.index, columns=ind_to_cols.keys())
        for ind, cols in ind_to_cols.items():
            if not cols:
                continue
            w_ind = w_norm[cols]
            den = w_ind.sum(axis=1)
            w_ind = w_ind.div(den.replace(0, np.nan), axis=0)
            if len(cols) > 0:
                w_ind = w_ind.fillna(1.0 / float(len(cols)))
            industry_rs_df[ind] = (rs_df[cols] * w_ind).sum(axis=1)
        try:
            logger.info(f"INDMOM industry_rs non-null={int(industry_rs_df.notna().sum().sum())}")
        except Exception:
            pass

        # Map per-instrument industry RS
        ind_map_aligned = ind_map.reindex(common_cols)
        rs_ind_for_inst = industry_rs_df[ind_map_aligned.values].set_axis(common_cols, axis=1)

        # Normalize weights within industry daily
        c_norm = pd.DataFrame(index=rs_df.index, columns=common_cols, dtype=float)
        for ind, cols in ind_to_cols.items():
            w_ind = w_raw[cols]
            den = w_ind.sum(axis=1)
            w_ind = w_ind.div(den.replace(0, np.nan), axis=0)
            if len(cols) > 0:
                w_ind = w_ind.fillna(1.0 / float(len(cols)))
            c_norm[cols] = w_ind

        indmom = rs_ind_for_inst - c_norm * rs_df
        try:
            nn_indmom = int((~(indmom.isna())).sum().sum())
            logger.info(f"INDMOM indmom non-null={nn_indmom}")
        except Exception:
            pass

        indmom.columns.name = "instrument"
        indmom.index.name = "datetime"
        indmom_long = indmom.stack().rename(self.out_col)
        out_df = indmom_long.astype(float).to_frame(self.out_col)
        try:
            logger.info(f"INDMOM out_series non-null={int(out_df[self.out_col].notna().sum())}")
        except Exception:
            pass
        # Join by MultiIndex to avoid reindex pitfalls
        try:
            df = df.drop(columns=[self.out_col], errors='ignore')
        except Exception:
            pass
        df = df.join(out_df, how='left')
        logger.info("IndustryMomentumProcessor finished...")
        return df


def _safe_log(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(series.clip(lower=eps))


def compute_size(mcap_last: pd.Series) -> pd.Series:
    last_caps = mcap_last.astype(float).replace([np.inf, -np.inf], np.nan)
    return -_safe_log(last_caps).fillna(last_caps.median())


def compute_momentum(returns: pd.DataFrame, mom_win: int = 504, gap: int = 21, halflife: int = 126) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    weights = np.exp(-np.log(2) * np.arange(mom_win) / max(halflife, 1))[::-1]
    weights /= weights.sum()
    log1p = np.log1p(returns.fillna(0))
    def _weighted_sum(x: np.ndarray) -> float:
        return float(np.nansum(weights * np.nan_to_num(x)))
    try:
        comp = log1p.rolling(mom_win).apply(_weighted_sum, raw=True, engine="numba")
    except Exception:
        comp = log1p.rolling(mom_win).apply(_weighted_sum, raw=True)
    last = comp.shift(gap).iloc[-1]
    return last.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_beta_resvol(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    beta_win: int = 252,
    resvol_win: int = 252,
    halflife_beta: int = 63,
) -> Tuple[pd.Series, pd.Series]:
    if returns.empty or market_returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    win = max(beta_win, resvol_win)
    rets_win = returns.iloc[-win:].fillna(0.0)
    mkt_win = market_returns.iloc[-win:].fillna(0.0)
    weights = np.exp(-np.log(2) * np.arange(win) / max(halflife_beta, 1))[::-1]
    weights /= weights.sum()
    rets_mean_w = rets_win.T.dot(weights)
    rets_center = rets_win - rets_mean_w
    mkt_mean_w = mkt_win.dot(weights)
    mkt_center = mkt_win - mkt_mean_w
    var_m = mkt_center.dot(weights * mkt_center)
    if var_m <= 0:
        zero = pd.Series(0.0, index=returns.columns)
        return zero, zero
    cov = rets_center.multiply(weights * mkt_center, axis=0).sum(axis=0)
    beta = cov / var_m
    fitted = pd.DataFrame(np.outer(mkt_win, beta), index=rets_win.index, columns=rets_win.columns)
    resid = rets_win - fitted
    resvol = resid.std()
    return beta, resvol


def compute_volatility(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    halflife_dastd: int = 42,
    win: int = 252,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    weights = np.exp(-np.log(2) * np.arange(win) / max(halflife_dastd, 1))[::-1]
    weights /= weights.sum()
    def _dastd_col(x: np.ndarray) -> float:
        return np.sqrt(np.nansum(weights * np.square(np.nan_to_num(x))))
    dastd_df = returns.rolling(win).apply(_dastd_col, raw=True)
    dastd = dastd_df.iloc[-1]
    cmra_window = min(win, 252)
    cum_log_ret = np.log1p(returns).cumsum().iloc[-cmra_window:]
    cmra = cum_log_ret.max() - cum_log_ret.min()
    _, hsigma = compute_beta_resvol(returns, market_returns, beta_win=win, resvol_win=win)
    vol = 0.74 * dastd + 0.16 * cmra + 0.10 * hsigma
    return vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_liquidity(turnover: pd.DataFrame, win_month: int = 21, win_quarter: int = 63, win_annual: int = 252) -> pd.Series:
    if turnover is None or turnover.empty:
        return pd.Series(dtype=float)
    stom = np.log(turnover.rolling(win_month).mean().iloc[-1] + 1e-6)
    stoq = np.log(turnover.rolling(win_quarter).mean().iloc[-1] + 1e-6)
    stoa = np.log(turnover.rolling(win_annual).mean().iloc[-1] + 1e-6)
    liq = 0.35 * stom + 0.35 * stoq + 0.30 * stoa
    return liq.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_earnings_yield(eps_trailing: pd.Series, eps_pred: pd.Series, price: pd.Series) -> pd.Series:
    if eps_trailing is None or price is None:
        return pd.Series(dtype=float)
    ep_trail = eps_trailing.astype(float) / price.replace(0, np.nan)
    ep_pred = (eps_pred.astype(float) / price.replace(0, np.nan)) if eps_pred is not None else pd.Series(0.0, index=ep_trail.index)
    ey = 0.5 * ep_trail + 0.5 * ep_pred
    return ey.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_cne6_exposures(
    returns: pd.DataFrame,
    mcap: pd.DataFrame,
    market_returns: pd.Series,
    turnover: Optional[pd.DataFrame] = None,
    eps_trailing: Optional[pd.DataFrame] = None,
    eps_pred: Optional[pd.DataFrame] = None,
    price: Optional[pd.DataFrame] = None,
    *,
    lookbacks: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    if mcap.empty:
        raise ValueError("mcap is required and must align with returns")
    if market_returns.empty:
        raise ValueError("market_returns is required and must align with returns")

    lb = {"beta": 252, "resvol": 252, "mom": 504, "mom_gap": 21, "vol_win": 252}
    if lookbacks is not None:
        lb.update({k: int(v) for k, v in lookbacks.items() if k in lb})

    rets = returns.sort_index().copy()
    caps = mcap.reindex_like(rets)
    mkt = market_returns.reindex(rets.index).fillna(0.0)

    size = compute_size(caps.iloc[-1])
    mom = compute_momentum(rets, mom_win=lb["mom"], gap=lb["mom_gap"])
    beta, resvol = compute_beta_resvol(rets, mkt, beta_win=lb["beta"], resvol_win=lb["resvol"])
    vol = compute_volatility(rets, mkt, win=lb["vol_win"])
    liq = compute_liquidity(turnover) if turnover is not None else pd.Series(0.0, index=rets.columns)
    if eps_trailing is not None and price is not None:
        eps_tr_last = eps_trailing.iloc[-1].reindex(rets.columns)
        eps_pred_last = eps_pred.iloc[-1].reindex(rets.columns) if eps_pred is not None else pd.Series(0.0, index=rets.columns)
        price_last = price.iloc[-1].reindex(rets.columns)
        ey = compute_earnings_yield(eps_tr_last, eps_pred_last, price_last)
    else:
        ey = pd.Series(0.0, index=rets.columns)

    exposures = pd.DataFrame({
        "SIZE": size, "BETA": beta, "MOM": mom, "RESVOL": resvol,
        "VOL": vol, "LIQ": liq, "EY": ey,
    })
    exposures = exposures.apply(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s)
    exposures = exposures.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return exposures