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


def _pivot_wide(df: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
    if col is None or col not in df.columns:
        return None
    ser = df[col]
    if not isinstance(ser.index, pd.MultiIndex) or ser.index.nlevels < 2:
        return None
    idx_names = list(ser.index.names)
    if "instrument" in idx_names:
        inst_level = idx_names.index("instrument")
    else:
        inst_level = ser.index.nlevels - 1  # assume last level is instrument when unnamed
    wide = ser.unstack(level=inst_level).sort_index()
    # Normalize index/column names for stable downstream ops
    wide.index.name = "datetime"
    wide.columns.name = "instrument"
    # Ensure numeric
    wide = wide.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return wide


class FundamentalProfileProcessor(Processor):
    """Load cn_profile fundamentals and align by announcement windows.

    - Source: data/others/cn_profile.csv
    - Columns required: ticker, period, ann_date, <factor columns>
    - Values are valid from current ann_date until next ann_date for the same ticker.
    """

    def __init__(self, csv_path: str = "data/others/cn_profile.csv", prefix: str = "", date_col: str = "ann_date"):
        super().__init__()
        self.csv_path = csv_path
        self.prefix = prefix
        self.date_col = date_col
        self._cache = None

    def fit(self, df: pd.DataFrame):
        return self

    def _load_profile(self) -> pd.DataFrame:
        if not self.csv_path:
            return pd.DataFrame()
        try:
            prof = pd.read_csv(self.csv_path)
        except (FileNotFoundError, ValueError):
            return pd.DataFrame()
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
        # If target df index has unnamed levels, skip reordering by names
        target_names = list(df.index.names)
        if any(n is None for n in target_names):
            return merged.sort_index()
        if list(merged.index.names) != target_names:
            merged = merged.reorder_levels(target_names).sort_index()
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
            "LD": ld,
            "TD": td,
            "BE": be,
            "TL": tl,
            "TA": ta,
            "SALES": sales,
            "COGS": cogs,
            "EARNINGS": earnings,
            "CASH": cash,
            "CFO": cfo,
            "CFI": cfi,
            "DA": da,
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
            df["ME"] = df["$market_cap"].astype(float)
        elif not me.empty:
            me = me.groupby(level=1).ffill()
            df["ME"] = me.reindex(df.index).astype(float)
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

    def _assign_quality_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        def col(name: str) -> pd.Series:
            return df.get(name, pd.Series(index=df.index, dtype=float))
        def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            val = a / b.replace(0, np.nan)
            return val.replace([np.inf, -np.inf], np.nan)
        ME = col("ME").astype(float)
        PE = col("PE").astype(float).fillna(0.0)
        LD = col("LD").astype(float).fillna(0.0)
        BE = col("BE").astype(float)
        TL = col("TL").astype(float)
        TA = col("TA").astype(float)
        SALES = col("SALES").astype(float)
        COGS = col("COGS").astype(float)
        EARN = col("EARNINGS").astype(float)
        CASH = col("CASH").astype(float).fillna(0.0)
        TD = col("TD").astype(float).fillna(0.0)
        CFO = col("CFO").astype(float).fillna(0.0)
        CFI = col("CFI").astype(float).fillna(0.0)
        DA = col("DA").astype(float).fillna(0.0)
        q_assign: Dict[str, pd.Series] = {}
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
        if "F_TOTAL_ASSETS_GROWTH" in df.columns and "Q_TAGR" not in q_assign:
            q_assign["Q_TAGR"] = df["F_TOTAL_ASSETS_GROWTH"]
        if "Q_MLEV" not in q_assign:
            q_assign["Q_MLEV"] = safe_div(ME + PE + LD, ME)
        if "Q_BLEV" not in q_assign:
            q_assign["Q_BLEV"] = safe_div(BE + PE + LD, ME)
        if "Q_DTOA" not in q_assign:
            q_assign["Q_DTOA"] = safe_div(TL, TA)
        GP = SALES - COGS
        if "Q_ATO" not in q_assign:
            q_assign["Q_ATO"] = safe_div(SALES, TA)
        if "Q_GP" not in q_assign:
            q_assign["Q_GP"] = safe_div(GP, TA)
        if "Q_GPM" not in q_assign:
            q_assign["Q_GPM"] = safe_div(GP, SALES)
        if "Q_ROA" not in q_assign:
            q_assign["Q_ROA"] = safe_div(EARN, TA)
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
        df = self._assign_quality_factors(df)
        # Attach industry info if available via IndustryProcessor
        try:
            df = IndustryProcessor()(df)
        except Exception:
            # Silently skip if industry mapping not available
            pass
        return df


class IndustryProcessor(Processor):
    """Attach industry code/name per ticker from a company facts file.

    - Source: data/others/cn_company_facts.csv (configurable)
    - Required columns: ticker, industry_code, industry
    - Values are static per ticker and will be attached to every date row.
    """

    def __init__(self, csv_path: str = "data/others/cn_company_facts.csv", prefix: str = "", code_col: str = "industry_code", name_col: str = "industry"):
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

        # Robust instrument level extraction when index names are missing
        if isinstance(df.index, pd.MultiIndex):
            idx_names = list(df.index.names)
            if "instrument" in idx_names:
                inst_level = idx_names.index("instrument")
            else:
                inst_level = df.index.nlevels - 1
            inst_index = df.index.get_level_values(inst_level).astype(str)
        else:
            inst_index = df.index.astype(str)
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
        industry_col: str = "INDUSTRY_CODE",
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
            return _pivot_wide(df, col)

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
        # Determine instrument level robustly
        idx_names_df = list(df.index.names) if isinstance(df.index, pd.MultiIndex) else []
        if "instrument" in idx_names_df:
            inst_level_df = idx_names_df.index("instrument")
        else:
            inst_level_df = (df.index.nlevels - 1) if isinstance(df.index, pd.MultiIndex) else 0
        ind_map = df[self.industry_col].groupby(level=inst_level_df).last()
        ind_map = ind_map.dropna().astype(str)
        if ind_map.empty:
            return df

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
            rs_col = np.convolve(log1p[:, j], w_flip, mode='valid')
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
        # Assign via reindex on values (robust to index name mismatches)
        series = out_df[self.out_col]
        desired_names = list(df.index.names)
        if all(n is not None for n in desired_names) and list(series.index.names) != desired_names:
            series = series.reorder_levels(desired_names).sort_index()
        df = df.copy()
        df[self.out_col] = series.reindex(df.index)

        logger.info("IndustryMomentumProcessor finished...")
        return df


def _safe_log(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(series.clip(lower=eps))


def compute_beta_resvol(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    beta_win: int = 252,
    resvol_win: int = 252,
    halflife_beta: int = 63,
    *,
    return_ts: bool = False,
    method: str = "ewm",
    min_periods: int = 30,
) -> Tuple[pd.Series, pd.Series]:
    """Compute CAPM beta and residual volatility.

    When return_ts is False (default), returns cross-sectional snapshot Series (index: instruments).
    When return_ts is True, returns time series DataFrames (index: datetime, columns: instruments).

    method: 'ewm' (exponentially weighted with halflife) or 'rolling' (simple rolling window).
    """
    if returns.empty or market_returns.empty:
        if return_ts:
            return pd.DataFrame(), pd.DataFrame()
        return pd.Series(dtype=float), pd.Series(dtype=float)

    rets = returns.sort_index().astype(float)
    mkt = market_returns.sort_index().astype(float)
    rets = rets.reindex(index=mkt.index).fillna(0.0)
    mkt = mkt.fillna(0.0)

    if not return_ts:
        win = max(beta_win, resvol_win)
        rets_win = rets.iloc[-win:]
        mkt_win = mkt.iloc[-win:]
        weights = np.exp(-np.log(2) * np.arange(win) / max(halflife_beta, 1))[::-1]
        weights /= weights.sum()
        rets_mean_w = rets_win.T.dot(weights)
        rets_center = rets_win - rets_mean_w
        mkt_mean_w = mkt_win.dot(weights)
        mkt_center = mkt_win - mkt_mean_w
        var_m = mkt_center.dot(weights * mkt_center)
        if var_m <= 0:
            zero = pd.Series(0.0, index=rets.columns)
            return zero, zero
        cov = rets_center.multiply(weights * mkt_center, axis=0).sum(axis=0)
        beta = cov / var_m
        fitted = pd.DataFrame(np.outer(mkt_win, beta), index=rets_win.index, columns=rets_win.columns)
        resid = rets_win - fitted
        resvol = resid.std()
        return beta, resvol

    # Time series mode
    beta_ts = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    resvol_ts = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)

    if method == "ewm":
        local_min = max(1, min(min_periods, len(mkt)))
        mkt_var = mkt.ewm(halflife=halflife_beta, min_periods=local_min, adjust=False).var()
        mkt_var = mkt_var.replace(0.0, np.nan)
        for col in rets.columns:
            cov = rets[col].ewm(halflife=halflife_beta, min_periods=local_min, adjust=False).cov(mkt)
            beta_col = cov / mkt_var
            beta_ts[col] = beta_col
            resid = rets[col] - beta_col * mkt
            resvol_ts[col] = resid.ewm(halflife=resvol_win // 3 if resvol_win else halflife_beta, min_periods=local_min, adjust=False).std(bias=False)
    else:
        # simple rolling window
        for col in rets.columns:
            local_min_beta = max(1, min(min_periods, beta_win))
            cov = rets[col].rolling(beta_win, min_periods=local_min_beta).cov(mkt)
            var = mkt.rolling(beta_win, min_periods=local_min_beta).var()
            beta_col = cov / var.replace(0.0, np.nan)
            beta_ts[col] = beta_col
            resid = rets[col] - beta_col * mkt
            local_min_res = max(1, min(min_periods, resvol_win))
            resvol_ts[col] = resid.rolling(resvol_win, min_periods=local_min_res).std()

    beta_ts = beta_ts.replace([np.inf, -np.inf], np.nan)
    resvol_ts = resvol_ts.replace([np.inf, -np.inf], np.nan)
    return beta_ts, resvol_ts



def compute_cne6_exposures_ts(
    returns: pd.DataFrame,
    mcap: pd.DataFrame,
    market_returns: pd.Series,
    turnover: Optional[pd.DataFrame] = None,
    eps_trailing: Optional[pd.DataFrame] = None,
    eps_pred: Optional[pd.DataFrame] = None,
    price: Optional[pd.DataFrame] = None,
    *,
    lookbacks: Optional[Dict[str, int]] = None,
    method_beta: str = "rolling",
    halflife_beta: int = 63,
) -> Dict[str, pd.DataFrame]:
    """Compute daily time series exposures for CNE6 factors.

    Returns a dict of DataFrames keyed by factor name (e.g., 'SIZE','BETA','MOM','RESVOL','VOL','LIQ','EY').
    Each DataFrame is indexed by datetime and has instruments as columns.
    """
    if returns.empty:
        return {}
    if mcap.empty:
        raise ValueError("mcap is required and must align with returns")
    if market_returns.empty:
        raise ValueError("market_returns is required and must align with returns")

    lb = {"beta": 252, "resvol": 252, "mom": 252, "mom_gap": 21, "vol_win": 252}
    if lookbacks is not None:
        lb.update({k: int(v) for k, v in lookbacks.items() if k in lb})

    rets = returns.sort_index().copy()
    caps = mcap.reindex_like(rets)
    # Align market series to returns index with unique dates
    mkt = market_returns.copy()
    if isinstance(mkt.index, pd.MultiIndex):
        mkt = mkt.droplevel(-1)
    if not mkt.index.is_unique:
        mkt = mkt.groupby(level=0).mean()
    mkt = mkt.reindex(rets.index).fillna(0.0)

    out: Dict[str, pd.DataFrame] = {}

    # SIZE: -log(ME) daily
    size_ts = -_safe_log(caps)
    out["SIZE"] = size_ts

    # MOM: weighted rolling of log(1+r), then shift by gap
    weights = np.exp(-np.log(2) * np.arange(lb["mom"]) / max(lb.get("halflife_mom", lb["mom"] // 4), 1))[::-1]
    weights /= weights.sum()
    log1p = np.log1p(rets.fillna(0.0)).to_numpy()
    mom_np = np.full(log1p.shape, np.nan, dtype=np.float32)
    w_flip = weights[::-1].astype(np.float32)
    n_rows = log1p.shape[0]
    for j in range(log1p.shape[1]):
        if n_rows >= lb["mom"]:
            comp = np.convolve(log1p[:, j], w_flip, mode='valid')
            # place from mom_win-1 onward
            mom_np[lb["mom"] - 1 :, j] = comp.astype(np.float32)
    mom_df = pd.DataFrame(mom_np, index=rets.index, columns=rets.columns)
    mom_df = mom_df.shift(lb["mom_gap"])  # apply gap
    out["MOM"] = mom_df

    # Beta & ResVol time series
    beta_ts, resvol_ts = compute_beta_resvol(
        rets, mkt, beta_win=lb["beta"], resvol_win=lb["resvol"], halflife_beta=halflife_beta, return_ts=True, method=method_beta
    )
    out["BETA"] = beta_ts
    out["RESVOL"] = resvol_ts

    # VOL time series
    # DASTD
    win = lb["vol_win"]
    w_vol = np.exp(-np.log(2) * np.arange(win) / max(int(win // 6) or 1, 1))[::-1]
    w_vol /= w_vol.sum()
    def _dastd_col(x: np.ndarray) -> float:
        return float(np.sqrt(np.nansum(w_vol * np.square(np.nan_to_num(x)))))
    dastd_ts = rets.rolling(win).apply(_dastd_col, raw=True)
    # CMRA approximated as rolling range of cumulative log returns
    cmra_window = min(win, 252)
    cum_log_ret = np.log1p(rets).cumsum()
    cmra_ts = cum_log_ret.rolling(cmra_window).apply(lambda a: float(np.nanmax(a) - np.nanmin(a)), raw=True)
    # Use ResVol as HSIGMA component
    hsigma_ts = resvol_ts.reindex_like(rets)
    vol_ts = 0.74 * dastd_ts + 0.16 * cmra_ts + 0.10 * hsigma_ts
    out["VOL"] = vol_ts

    # LIQ family time series: STOM/STOQ/STOA/ATVR, normalized by market cap when available
    if turnover is not None and not turnover.empty:
        # Input is turnover rate already; use directly
        rate_df = turnover.astype(float)

        # Use full-window semantics so different windows produce distinct signals
        # Use log1p so zero rates map to 0 (not a large negative constant)
        mean_21 = rate_df.clip(lower=0).rolling(21).mean()
        mean_63 = rate_df.clip(lower=0).rolling(63).mean()
        mean_252 = rate_df.clip(lower=0).rolling(252).mean()
        stom = np.log1p(mean_21)
        stoq = np.log1p(mean_63)
        stoa = np.log1p(mean_252)
        out["STOM"] = stom
        out["STOQ"] = stoq
        out["STOA"] = stoa

        # ATVR from the same base rate
        atvr = rate_df.ewm(halflife=63, min_periods=30, adjust=False).mean() * 252.0
        out["ATVR"] = atvr

        # Composite LIQ with adaptive weights when some windows are unavailable
        w_stom = stom.notna().astype(float) * 0.35
        w_stoq = stoq.notna().astype(float) * 0.35
        w_stoa = stoa.notna().astype(float) * 0.30
        w_sum = (w_stom + w_stoq + w_stoa)
        num = stom.fillna(0.0) * 0.35 + stoq.fillna(0.0) * 0.35 + stoa.fillna(0.0) * 0.30
        liq_ts = num.div(w_sum.replace(0.0, np.nan))
        out["LIQ"] = liq_ts

    # EY time series
    if eps_trailing is not None and price is not None:
        eps_tr_ts = eps_trailing.reindex_like(price).astype(float)
        eps_pr_ts = eps_pred.reindex_like(price).astype(float) if eps_pred is not None else pd.DataFrame(0.0, index=price.index, columns=price.columns)
        price_ts = price.astype(float).replace(0.0, np.nan)
        ep_trail = eps_tr_ts / price_ts
        ep_pred = eps_pr_ts / price_ts
        ey_ts = 0.5 * ep_trail + 0.5 * ep_pred
        out["EY"] = ey_ts

    # STREV (short-term reversal): EW sum of daily returns over 21d with halflife=5
    try:
        win_strev = 21
        hl_strev = 5
        w_strev = np.exp(-np.log(2) * np.arange(win_strev) / max(hl_strev, 1))[::-1]
        w_strev /= w_strev.sum()
        rets_np = rets.fillna(0.0).to_numpy()
        strev_np = np.full(rets_np.shape, np.nan, dtype=np.float32)
        w_flip = w_strev[::-1].astype(np.float32)
        for j in range(rets_np.shape[1]):
            comp = np.convolve(rets_np[:, j], w_flip, mode='valid')
            strev_np[win_strev - 1 :, j] = comp.astype(np.float32)
        strev_ts = pd.DataFrame(strev_np, index=rets.index, columns=rets.columns)
        out["STREV"] = strev_ts
    except Exception as ex:
        print(f"Failed to compute STREV: {ex}")
        pass

    # SEASON (seasonality): average of prior 5 years' same-month returns, broadcast to days within the month
    try:
        nyears = 5
        if price is not None and not price.empty:
            m_last = price.resample("M").last()
            m_ret = m_last.pct_change()
        else:
            m_ret = (1.0 + rets).resample("M").apply(np.prod) - 1.0
        season_m = None
        for i in range(1, nyears + 1):
            shifted = m_ret.shift(12 * i)
            season_m = shifted if season_m is None else (season_m + shifted)
        if season_m is None:
            season_m = m_ret * np.nan
        season_m = season_m / nyears
        # Map month-end seasonality to daily dates by reindexing on month_end of daily index
        month_end_idx = rets.index.to_period("M").to_timestamp("M")
        season_daily = season_m.reindex(month_end_idx)
        season_daily.index = rets.index
        out["SEASON"] = season_daily
    except Exception as ex:
        print(f"Failed to compute SEASON: {ex}")
        pass

    # Clean up infinities
    for k, df_ts in list(out.items()):
        if df_ts is None or df_ts.empty:
            continue
        out[k] = df_ts.replace([np.inf, -np.inf], np.nan)

    return out

class BarraFactorProcessor(Processor):
    """End-to-end processor to attach F_/Q_/Industry/CNE6/B_INDMOM in one pass.

    - Reuses FundamentalProfileProcessor, BarraFundamentalQualityProcessor, IndustryProcessor
    - Uses compute_cne6_exposures with robust wide pivot via _pivot_wide
    - Computes industry momentum via IndustryMomentumProcessor (robust write-back)
    """

    def __init__(
        self,
        *,
        profile_csv_path: str = "data/others/cn_profile.csv",
        facts_csv_path: str = "data/others/cn_company_facts.csv",
        profile_prefix: str = "F_",
        market_col: str = "$market_ret",
        turnover_col: str = "$turnover",
        eps_trailing_col: Optional[str] = None,
        eps_pred_col: Optional[str] = None,
        price_col: str = "$close",
        date_col: str = "ann_date",
        lookbacks: Optional[Dict[str, int]] = None,
        indmom_window: int = 126,
        indmom_halflife: int = 21,
        indmom_out: str = "B_INDMOM",
        debug: bool = False,
    ):
        super().__init__()
        self.profile_csv_path = profile_csv_path
        self.facts_csv_path = facts_csv_path
        self.profile_prefix = profile_prefix
        self.market_col = market_col
        self.turnover_col = turnover_col
        self.eps_trailing_col = eps_trailing_col
        self.eps_pred_col = eps_pred_col
        self.price_col = price_col
        self.date_col = date_col
        self.lookbacks = lookbacks or {}
        self.indmom_window = int(indmom_window)
        self.indmom_halflife = int(indmom_halflife)
        self.indmom_out = indmom_out
        self.debug = debug

    def fit(self, df: pd.DataFrame):
        return self

    # (_attach_exposures removed as unused)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        assert isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2, "Input df must be MultiIndex (datetime, instrument)"

        # 1) Fundamentals
        df = FundamentalProfileProcessor(csv_path=self.profile_csv_path, prefix="")(df)

        # 2) Quality factors already computed within FundamentalProfileProcessor

        # 3) Industry mapping
        df = IndustryProcessor(csv_path=self.facts_csv_path)(df)

        # 4) CNE6 exposures (robust pivot)
        returns = _pivot_wide(df, "$return")
        if returns is None:
            close = _pivot_wide(df, self.price_col)
            if close is not None:
                returns = close.pct_change()
        mcap = _pivot_wide(df, "$market_cap")
        market = _pivot_wide(df, self.market_col)
        market_series = market.iloc[:, 0] if market is not None else None
        turnover = _pivot_wide(df, self.turnover_col) if self.turnover_col else None
        eps_trailing = _pivot_wide(df, self.eps_trailing_col) if self.eps_trailing_col else None
        eps_pred = _pivot_wide(df, self.eps_pred_col) if self.eps_pred_col else None
        price = _pivot_wide(df, self.price_col) if self.price_col else None

        if returns is not None and mcap is not None and market_series is not None:
            # Compute daily time-series exposures
            expos_ts = compute_cne6_exposures_ts(
                returns=returns,
                mcap=mcap,
                market_returns=market_series,
                turnover=turnover,
                eps_trailing=eps_trailing,
                eps_pred=eps_pred,
                price=price,
                lookbacks=self.lookbacks,
                method_beta="rolling",
                halflife_beta=63,
            )
            # Write back each factor daily
            for key, ts_df in expos_ts.items():
                if ts_df is None or ts_df.empty:
                    continue
                ts_df.index.name = "datetime"
                ts_df.columns.name = "instrument"
                ser = ts_df.stack().rename(f"B_{key}")
                if any(n is None for n in df.index.names) or list(ser.index.names) != list(df.index.names):
                    try:
                        ser = ser.reorder_levels(df.index.names).sort_index()
                    except Exception:
                        pass
                df[f"B_{key}"] = ser.reindex(df.index)

        # 5) Industry momentum (robust write-back & alignment already handled inside)
        df = IndustryMomentumProcessor(
            return_col="$return",
            close_col=self.price_col,
            mcap_col="$market_cap",
            industry_col="INDUSTRY_CODE",
            window=self.indmom_window,
            halflife=self.indmom_halflife,
            out_col=self.indmom_out,
            debug=self.debug,
        )(df)

        logger.info("BarraFactorProcessor finished.")
        return df