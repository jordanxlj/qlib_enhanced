# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np

from qlib.data.dataset.processor import Processor


class BarraCNE6Processor(Processor):
    """Processor to compute and attach Barra-style exposures via CNE6 functions.

    It expects the input DataFrame to be MultiIndex (datetime, instrument),
    with columns including at least:
      - "$return" (or precomputed returns)
      - "$mkt_cap"
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
        turnover_col: Optional[str] = None,
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
        mcap = pivot("$mkt_cap")
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
                df = df.join(pd.DataFrame(assign, index=df.index))
        return df


class FundamentalProfileProcessor(Processor):
    """Load cn_profile fundamentals and align by announcement windows.

    - Source: data/ann_report/cn_profile.csv
    - Columns required: ticker, period, ann_date, <factor columns>
    - Values are valid from current ann_date until next ann_date for the same ticker.
    """

    def __init__(self, csv_path: str = "data/ann_report/cn_profile.csv", prefix: str = "F_", date_col: str = "ann_date"):
        super().__init__()
        self.csv_path = csv_path
        self.prefix = prefix
        self.date_col = date_col
        self._cache = None

    def fit(self, df: pd.DataFrame):
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        assert isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2, "Input df must be MultiIndex (datetime, instrument)"

        if self._cache is None:
            prof = pd.read_csv(self.csv_path)
            # normalize cols
            prof.columns = [str(c).strip().lower() for c in prof.columns]
            assert "ticker" in prof.columns and self.date_col in prof.columns, "CSV must include ticker and ann_date"
            # robust ann_date parsing: accept 'YYYY-MM-DD' or numeric 'YYYYMMDD'
            raw_ad = prof[self.date_col]
            ad: pd.Series
            try:
                import pandas.api.types as ptypes
                if ptypes.is_numeric_dtype(raw_ad):
                    vals = pd.to_numeric(raw_ad, errors="coerce").astype("Int64")
                    # detect YYYYMMDD range
                    mask = (vals >= 19000101) & (vals <= 21000101)
                    if mask.mean() > 0.5:
                        ad = pd.to_datetime(vals.astype(str), format="%Y%m%d", errors="coerce")
                    else:
                        ad = pd.to_datetime(raw_ad, errors="coerce")
                else:
                    s = raw_ad.astype(str).str.strip()
                    if (s.str.fullmatch(r"\d{8}").mean() > 0.5):
                        ad = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
                    else:
                        ad = pd.to_datetime(s, errors="coerce")
            except Exception:
                ad = pd.to_datetime(raw_ad, errors="coerce")
            prof[self.date_col] = ad
            # normalize ticker to Qlib format, e.g., 300058.SZ -> SZ300058
            def _norm_ticker(x: str) -> str:
                s = str(x).strip().upper()
                if "." in s:
                    base, exch = s.split(".")
                    base = base[-6:].zfill(6)
                    if exch in ("SZ", "SH", "BJ"):
                        return f"{exch}{base}"
                return s
            prof["ticker"] = prof["ticker"].map(_norm_ticker)
            # Coerce potential numeric fields to numeric
            key_cols = {"ticker", self.date_col, "period"}
            factor_cols_all = [c for c in prof.columns if c not in key_cols]
            ratio_like = {
                "gross_margin",
                "operating_margin",
                "net_margin",
                "return_on_equity",
                "return_on_assets",
                "asset_turnover",
                "fixed_asset_turnover",
                "debt_to_equity",
                "debt_to_assets",
                "return_on_invested_capital",
                "working_capital_turnover",
            }
            for c in factor_cols_all:
                series = prof[c]
                if series.dtype == object:
                    cleaned = (
                        series.astype(str)
                        .str.strip()
                        .str.replace('%', '', regex=False)
                        .str.replace(',', '', regex=False)
                        # replace common unicode minus/dash with hyphen-minus
                        .str.replace('\u2212', '-', regex=False)  # −
                        .str.replace('\u2013', '-', regex=False)  # –
                        .str.replace('\u2014', '-', regex=False)  # —
                    )
                    col_series = pd.to_numeric(cleaned, errors="coerce")
                else:
                    col_series = pd.to_numeric(series, errors="coerce")
                # scale percentage only for ratio-like columns
                if c in ratio_like:
                    finite = col_series.replace([np.inf, -np.inf], np.nan).dropna()
                    if not finite.empty and finite.quantile(0.95) > 1.5:
                        col_series = col_series / 100.0
                prof[c] = col_series

            prof.sort_values(["ticker", self.date_col], inplace=True)
            self._cache = prof
        else:
            prof = self._cache

        # Robustly extract level names
        idx_names = list(df.index.names)
        # Default MultiIndex order is (datetime, instrument)
        if "instrument" in idx_names:
            inst_level = idx_names.index("instrument")
        else:
            inst_level = 1 if len(idx_names) > 1 else 0
        if "datetime" in idx_names:
            date_level = idx_names.index("datetime")
        else:
            date_level = 0

        trade_dates = df.index.get_level_values(date_level).unique().sort_values()
        codes = df.index.get_level_values(inst_level).unique()
        code_set = set(map(str, codes))

        key_cols = {"ticker", self.date_col, "period"}
        factor_cols = [c for c in prof.columns if c not in key_cols]
        if not factor_cols:
            return df

        frames = []
        for ticker, g in prof.groupby("ticker"):
            if str(ticker) not in code_set:
                continue
            ts = g.set_index(self.date_col)[factor_cols].sort_index()
            # Drop duplicate announcement dates per ticker, keep the latest record
            if ts.index.has_duplicates:
                ts = ts[~ts.index.duplicated(keep="last")]
            # Align to trade_dates then forward-fill; avoids monotonic index requirement
            aligned = pd.DataFrame(index=trade_dates).join(ts, how='left')
            aligned = aligned.ffill()
            aligned["instrument"] = ticker
            aligned["datetime"] = aligned.index
            frames.append(aligned)
        if not frames:
            return df
        merged = pd.concat(frames, axis=0)
        merged.set_index(["datetime", "instrument"], inplace=True)
        merged.sort_index(inplace=True)
        # Ensure index level order matches the input df to allow reindex alignment
        try:
            if list(merged.index.names) != list(df.index.names):
                merged = merged.reorder_levels(df.index.names).sort_index()
        except Exception:
            pass

        # Debug: basic info about merged fundamentals
        try:
            print(
                "[FundamentalProfileProcessor] merged shape:", merged.shape,
                "; columns contains 'return_on_equity':",
                ("return_on_equity" in merged.columns),
            )
            if "return_on_equity" in merged.columns:
                nonnull = int(merged["return_on_equity"].notna().sum())
                print("[FundamentalProfileProcessor] return_on_equity non-null in merged:", nonnull)
                ex = merged[["return_on_equity"]].dropna().head(5)
                if not ex.empty:
                    print("[FundamentalProfileProcessor] sample merged ROE:\n", ex.to_string())
        except Exception:
            pass

        # Batch build fundamental columns to reduce fragmentation
        assign_map = {}
        for col in factor_cols:
            out = f"{self.prefix}{col.upper()}"
            ser = merged[col]
            # forward fill across dates per instrument to ensure persistence until next announcement
            ser = ser.groupby(level=1).ffill()
            assign_map[out] = ser.reindex(df.index)

        # Derive core fields required by quality processor from cn_profile schema
        get = lambda c: merged[c] if c in merged.columns else pd.Series(index=merged.index, dtype=float)
        # Base components
        me = get("market_cap")  # fallback only; primary source is $mkt_cap from data loader
        ld = get("long_term_debt")
        td = get("total_debt")
        st_debt = get("short_term_debt")
        if td.empty or td.isna().all():
            td = (ld.fillna(0.0) + st_debt.fillna(0.0))
        be = get("shareholders_equity")
        tl = get("total_liabilities")
        ta = get("total_assets")
        sales = get("revenue")
        gross = get("gross_profit")
        cogs = (sales - gross) if (not sales.empty and not gross.empty) else pd.Series(index=merged.index, dtype=float)
        earnings = get("net_income")
        cash = get("cash_and_equivalents")
        cfo = get("operating_cash_flow")
        capex = get("capital_expenditure")
        cfi = -capex if not capex.empty else pd.Series(index=merged.index, dtype=float)
        da = get("depreciation_and_amortization")

        # Improve total debt fallback: if both components missing, keep NaN (not 0)
        if (td is None or td.empty) or td.isna().all():
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
        # Batch-assign derived
        for out, ser in derived.items():
            if ser is None or ser.empty:
                continue
            ser = ser.groupby(level=1).ffill()
            assign_map[out] = ser.reindex(df.index).astype(float)

        # Apply batch assignment for fundamentals
        if assign_map:
            df = df.join(pd.DataFrame(assign_map))

        # Set F_ME from $mkt_cap (from market_cap.day.bin) instead of cn_profile; fallback to profile if missing
        if "$mkt_cap" in df.columns:
            try:
                df["F_ME"] = df["$mkt_cap"].astype(float)
            except Exception:
                df["F_ME"] = df["$mkt_cap"]
        elif me is not None and not me.empty:
            me = me.groupby(level=1).ffill()
            df["F_ME"] = me.reindex(df.index).astype(float)

        # Prefer using ratio fields directly if provided in cn_profile.csv
        ratio_map = {
            "F_DTOA_RATIO": "debt_to_assets",
            "F_DEBT_TO_EQUITY": "debt_to_equity",
            "F_ASSET_TURNOVER": "asset_turnover",
            "F_FIXED_ASSET_TURNOVER": "fixed_asset_turnover",
            "F_GROSS_MARGIN": "gross_margin",
            "F_OPERATING_MARGIN": "operating_margin",
            "F_NET_MARGIN": "net_margin",
            "F_ROA_RATIO": "return_on_assets",
            "F_ROE_RATIO": "return_on_equity",
            # common aliases
            "F_RETURN_ON_EQUITY": "return_on_equity",
            "F_RETURN_ON_ASSETS": "return_on_assets",
            # handle raw column names as-is (lowercased earlier)
            # e.g., if the CSV already uses 'return_on_equity' we will pick it directly
        }
        # Explicitly map key ratio columns first (min-scope change)
        if "return_on_equity" in merged.columns:
            ser = merged["return_on_equity"].groupby(level="instrument").ffill()
            # align by exact (datetime, instrument) index
            ser = ser.reindex(merged.index)
            df["F_ROE_RATIO"] = ser.reindex(df.index).astype(float)
        if "return_on_assets" in merged.columns:
            ser = merged["return_on_assets"].groupby(level="instrument").ffill()
            ser = ser.reindex(merged.index)
            df["F_ROA_RATIO"] = ser.reindex(df.index).astype(float)

        try:
            if "F_ROE_RATIO" in df.columns:
                print(
                    "[FundamentalProfileProcessor] F_ROE_RATIO non-null after assign:",
                    int(df["F_ROE_RATIO"].notna().sum()),
                )
                ex = df[["F_ROE_RATIO"]].dropna().head(5)
                if not ex.empty:
                    print("[FundamentalProfileProcessor] sample F_ROE_RATIO rows:\n", ex.to_string())
        except Exception:
            pass

        ratio_assign = {}
        for out, colname in ratio_map.items():
            # colname is lower-case CSV field; merged columns keep original CSV names
            # which we lowercased earlier when reading into cache.
            if colname in merged.columns:
                ser = merged[colname]
            else:
                # try a few alias fallbacks
                aliases = {
                    "return_on_equity": ["roe", "roe_ratio", "return_on_equity(%)"],
                    "return_on_assets": ["roa", "roa_ratio", "return_on_assets(%)"],
                }
                ser = None
                if colname in aliases:
                    for a in aliases[colname]:
                        if a in merged.columns:
                            ser = merged[a]
                            break
                if ser is None:
                    continue
            ser = ser.groupby(level=1).ffill()
            ratio_assign[out] = ser.reindex(df.index).astype(float)
        if ratio_assign:
            ratio_df = pd.DataFrame(ratio_assign)
            # Overwrite existing columns to avoid overlap error
            for c in ratio_df.columns:
                df[c] = ratio_df[c]
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
            # Keep NaNs to avoid creating constant zeros that harm IC
            return val.replace([np.inf, -np.inf], np.nan)

        # Basic components
        ME = col("F_ME").astype(float)
        # Treat missing components that are summed as zeros to avoid NaN propagation
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

        # Prefer ratio columns if available (directly from cn_profile)
        q_assign = {}
        if "F_DTOA_RATIO" in df.columns:
            q_assign["Q_DTOA"] = df["F_DTOA_RATIO"]
        if "F_ASSET_TURNOVER" in df.columns:
            q_assign["Q_ATO"] = df["F_ASSET_TURNOVER"]
        if "F_GROSS_MARGIN" in df.columns:
            q_assign["Q_GPM"] = df["F_GROSS_MARGIN"]
        if "F_ROA_RATIO" in df.columns:
            q_assign["Q_ROA"] = df["F_ROA_RATIO"]

        # Leverage (fallback to totals when ratios not provided)
        if "Q_MLEV" not in q_assign and "Q_MLEV" not in df.columns:
            q_assign["Q_MLEV"] = safe_div(ME + PE + LD, ME)
        if "Q_BLEV" not in q_assign and "Q_BLEV" not in df.columns:
            q_assign["Q_BLEV"] = safe_div(BE + PE + LD, ME)
        if "Q_DTOA" not in q_assign and "Q_DTOA" not in df.columns:
            q_assign["Q_DTOA"] = safe_div(TL, TA)

        # Profitability & turnover
        GP = (SALES - COGS)
        if "Q_ATO" not in q_assign and "Q_ATO" not in df.columns:
            q_assign["Q_ATO"] = safe_div(SALES, TA)
        if "Q_GP" not in q_assign and "Q_GP" not in df.columns:
            q_assign["Q_GP"] = safe_div(GP, TA)
        if "Q_GPM" not in q_assign and "Q_GPM" not in df.columns:
            q_assign["Q_GPM"] = safe_div(GP, SALES)
        if "Q_ROA" not in q_assign and "Q_ROA" not in df.columns:
            q_assign["Q_ROA"] = safe_div(EARN, TA)

        # Accruals: balance sheet and cashflow versions
        NOA = (TA - CASH) - (TL - TD)
        dNOA = NOA.groupby(level=1).diff()
        ACCR_BS = dNOA - DA
        if "Q_ABS" not in q_assign and "Q_ABS" not in df.columns:
            q_assign["Q_ABS"] = -safe_div(ACCR_BS, TA)

        ACCR_CF = EARN - (CFO + CFI) + DA
        if "Q_ACF" not in q_assign and "Q_ACF" not in df.columns:
            q_assign["Q_ACF"] = -safe_div(ACCR_CF, TA)

        if q_assign:
            df = df.join(pd.DataFrame(q_assign))

        return df

def _safe_log(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(series.clip(lower=eps))


def compute_size(mcap_last: pd.Series) -> pd.Series:
    last_caps = mcap_last.astype(float).replace([np.inf, -np.inf], np.nan)
    return -_safe_log(last_caps).fillna(last_caps.median())


def compute_momentum(returns: pd.DataFrame, mom_win: int = 504, gap: int = 21, halflife: int = 126) -> pd.Series:
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    weights = np.exp(-np.log(2) * np.arange(mom_win) / max(int(halflife), 1))[::-1]
    weights = weights / weights.sum()
    log1p = returns.applymap(lambda r: np.log(1.0 + (0.0 if pd.isna(r) else r)))
    comp = log1p.rolling(int(mom_win)).apply(lambda x: float(np.nansum(weights * np.nan_to_num(x))), raw=False)
    last = comp.shift(int(gap)).iloc[-1] if comp.shape[0] > 0 else pd.Series(0.0, index=returns.columns)
    return last.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_beta_resvol(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    beta_win: int = 252,
    resvol_win: int = 252,
    halflife_beta: int = 63,
) -> Tuple[pd.Series, pd.Series]:
    if returns is None or returns.empty or market_returns is None or market_returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    win = max(int(beta_win), int(resvol_win))
    rets_win = returns.iloc[-win:].fillna(0.0)
    mkt_win = market_returns.iloc[-win:].fillna(0.0)
    weights = np.exp(-np.log(2) * np.arange(win) / max(int(halflife_beta), 1))[::-1]
    weights = weights / weights.sum()
    rets_mean_w = pd.Series(np.asarray(rets_win.T @ weights).reshape(-1), index=rets_win.columns)
    rets_center = rets_win - rets_mean_w
    mkt_mean_w = float(mkt_win @ weights)
    mkt_center = mkt_win - mkt_mean_w
    var_m = float(mkt_center @ (weights * mkt_center))
    if var_m <= 0:
        zero = pd.Series(0.0, index=returns.columns)
        return zero, zero
    cov = (rets_center.multiply(weights * mkt_center, axis=0)).sum(axis=0)
    beta = cov / var_m
    fitted = pd.DataFrame(np.outer(mkt_win.values, beta.values), index=rets_win.index, columns=rets_win.columns)
    resid = rets_win - fitted
    resvol = resid.std(axis=0)
    return beta, resvol


def compute_volatility(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    halflife_dastd: int = 42,
    win: int = 252,
) -> pd.Series:
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    win = int(win)
    weights = np.exp(-np.log(2) * np.arange(win) / max(int(halflife_dastd), 1))[::-1]
    weights = weights / weights.sum()
    def _dastd_col(x: np.ndarray) -> float:
        return float(np.sqrt(np.nansum(weights * np.square(np.nan_to_num(x)))))
    dastd_df = returns.rolling(win).apply(_dastd_col, raw=True)
    dastd = dastd_df.iloc[-1] if dastd_df.shape[0] > 0 else pd.Series(0.0, index=returns.columns)
    cmra_window = min(win, 252)
    cum_log_ret = np.log(1.0 + returns).cumsum().iloc[-cmra_window:]
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
    if returns is None or returns.empty:
        return pd.DataFrame()
    if mcap is None or mcap.empty:
        raise ValueError("mcap is required and must align with returns")
    if market_returns is None or market_returns.empty:
        raise ValueError("market_returns is required and must align with returns")

    lb = {"beta": 252, "resvol": 252, "mom": 504, "mom_gap": 21, "vol_win": 252}
    if lookbacks is not None:
        lb.update({k: int(v) for k, v in lookbacks.items() if k in lb})

    rets = returns.sort_index().copy()
    caps = mcap.reindex_like(rets)
    mkt = market_returns.reindex(rets.index).fillna(0.0)

    size = compute_size(caps.iloc[-1])
    mom = compute_momentum(rets, mom_win=lb["mom"], gap=lb["mom_gap"]) if len(rets) else pd.Series(0.0, index=rets.columns)
    beta, resvol = (
        compute_beta_resvol(rets, mkt, beta_win=lb["beta"], resvol_win=lb["resvol"]) if len(rets)
        else (pd.Series(0.0, index=rets.columns), pd.Series(0.0, index=rets.columns))
    )
    vol = compute_volatility(rets, mkt, win=lb["vol_win"]) if len(rets) else pd.Series(0.0, index=rets.columns)
    liq = compute_liquidity(turnover) if turnover is not None else pd.Series(0.0, index=rets.columns)
    if eps_trailing is not None and price is not None:
        eps_tr_last = (eps_trailing.iloc[-1] if isinstance(eps_trailing, pd.DataFrame) else eps_trailing).reindex(rets.columns)
        eps_pred_last = (eps_pred.iloc[-1] if isinstance(eps_pred, pd.DataFrame) else eps_pred).reindex(rets.columns) if eps_pred is not None else pd.Series(0.0, index=rets.columns)
        price_last = (price.iloc[-1] if isinstance(price, pd.DataFrame) else price).reindex(rets.columns)
        ey = compute_earnings_yield(eps_tr_last, eps_pred_last, price_last)
    else:
        ey = pd.Series(0.0, index=rets.columns)

    exposures = pd.DataFrame({
        "SIZE": size, "BETA": beta, "MOM": mom, "RESVOL": resvol,
        "VOL": vol, "LIQ": liq, "EY": ey,
    })
    exposures = exposures.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    exposures = exposures.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return exposures
