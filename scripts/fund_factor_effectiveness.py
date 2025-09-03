# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Quick test script to evaluate effectiveness of fundamental (quality) factors.

It computes daily cross-sectional IC (Spearman rank correlation) between
selected fundamental factors and next-day returns over a recent window.

Usage (from repo root after preparing data):
    python scripts/fund_factor_effectiveness.py
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import qlib
from qlib.data import D

from qlib.contrib.data.barra_processor import (
    FundamentalProfileProcessor,
    BarraFundamentalQualityProcessor,
    IndustryMomentumProcessor,
)


def _assert_factor_alignment(df: pd.DataFrame, csv_path: str, raw_ticker: str, dates: List[str], factor_name: str, raw_col: str) -> None:
    """Assert that factor_name equals raw cn_profile value (ffilled) around given dates.

    - Normalize ticker to Qlib format (e.g., 000001.SZ -> SZ000001).
    - Parse and clean raw value from cn_profile.csv (handle %, commas, Unicode minus, and /100 if needed).
    - For each target date, pick the nearest prev and next trading dates present in df, and assert equality
      between factor_name and the last announced value as of that trading date.
    """
    # Normalize ticker to qlib instrument format
    s = raw_ticker.strip().upper()
    if "." in s:
        base, exch = s.split(".")
        base = base[-6:].zfill(6)
        inst = f"{exch}{base}"
    else:
        inst = s

    # Load cn_profile.csv
    try:
        prof = pd.read_csv(csv_path)
    except Exception as e:
        raise AssertionError(f"Failed to read {csv_path}: {e}")
    prof.columns = [c.lower() for c in prof.columns]

    # Parse ann_date robustly
    ad_raw = prof.get("ann_date")
    if ad_raw is None:
        raise AssertionError("cn_profile.csv missing ann_date column")
    try:
        vals = pd.to_numeric(ad_raw, errors="coerce")
        if vals.notna().mean() > 0.5 and (vals.between(19000101, 21001231).mean() > 0.5):
            ad = pd.to_datetime(vals.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            ad = pd.to_datetime(ad_raw.astype(str).str.strip(), errors="coerce")
    except Exception:
        ad = pd.to_datetime(ad_raw, errors="coerce")
    prof["ann_date"] = ad.dt.normalize()

    # Normalize tickers to Qlib format
    def _norm_ticker(x: str) -> str:
        t = str(x).strip().upper()
        if "." in t:
            base, exch = t.split(".")
            base = base[-6:].zfill(6)
            if exch in ("SZ", "SH", "BJ"):
                return f"{exch}{base}"
        return t

    prof["ticker"] = prof.get("ticker", prof.get("code", prof.get("symbol", ""))).map(_norm_ticker)
    prof = prof[prof["ticker"] == inst].copy()
    if prof.empty:
        raise AssertionError(f"No cn_profile rows for {inst}")

    # Check if raw_col exists
    if raw_col not in prof.columns:
        raise AssertionError(f"{raw_col} column not found in cn_profile.csv")

    # Clean numeric values; handle %, commas, Unicode minus; scale if looks like percentage 0-100
    cleaned = (
        prof[raw_col]
        .astype(str)
        .str.strip()
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace('\u2212', '-', regex=False)
        .str.replace('\u2013', '-', regex=False)
        .str.replace('\u2014', '-', regex=False)
    )
    raw_vals = pd.to_numeric(cleaned, errors="coerce")
    finite = raw_vals.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty and finite.quantile(0.95) > 1.5:
        raw_vals = raw_vals / 100.0

    # Build last-wins series by announcement date
    s_raw = (
        pd.Series(raw_vals.values, index=prof["ann_date"]).sort_index()
        .loc[lambda x: x.index.notna()]
    )
    s_raw = s_raw[~s_raw.index.duplicated(keep="last")]

    print(f"Announcement dates for {inst} ({raw_col}):")
    print(s_raw.to_string())

    # Helper to get expected ffilled raw value for a given date
    def expected_value_on(date_ts: pd.Timestamp) -> float:
        if s_raw.empty:
            return np.nan
        # T+1 effective: use last announcement strictly before the trading date
        pos = s_raw.index.searchsorted(pd.Timestamp(date_ts).normalize(), side="left") - 1
        if pos < 0 or pos >= len(s_raw):
            return np.nan
        val = float(s_raw.iloc[pos])
        return val

    # Prepare trading dates and select instrument slice
    if "instrument" not in df.index.names or "datetime" not in df.index.names:
        raise AssertionError("Dataframe must have a MultiIndex with levels ('datetime', 'instrument').")
    inst_values = df.index.get_level_values("instrument").unique()
    if inst not in set(inst_values):
        raise AssertionError(f"Instrument {inst} not present in processed dataframe.")
    s_fac = df.xs(inst, level="instrument")
    if factor_name not in s_fac.columns:
        raise AssertionError(f"{factor_name} not present in processed dataframe.")
    dts = s_fac.index.unique().sort_values()

    # For each target date, check prev and next trading dates
    for d in dates:
        tgt = pd.Timestamp(d)
        # prev trading day
        idx_prev = dts.searchsorted(tgt, side="right") - 1
        if 0 <= idx_prev < len(dts):
            dt_prev = dts[idx_prev]
            fac_prev = s_fac.at[dt_prev, factor_name]
            exp_prev = expected_value_on(dt_prev)
            if pd.isna(fac_prev) and pd.isna(exp_prev):
                pass
            else:
                assert np.isfinite(exp_prev), f"Expected raw value is NaN for prev {dt_prev}"
                assert np.isclose(float(fac_prev), float(exp_prev), atol=1e-4, rtol=0.0), (
                    f"Mismatch prev {inst} {dt_prev.date()}: {factor_name}={fac_prev}, raw={exp_prev}"
                )
                print(f"Checking prev {dt_prev.date()}: fac={fac_prev}, exp={exp_prev}")
        # next trading day
        idx_next = dts.searchsorted(tgt, side="left")
        if 0 <= idx_next < len(dts):
            dt_next = dts[idx_next]
            fac_next = s_fac.at[dt_next, factor_name]
            exp_next = expected_value_on(dt_next)
            if pd.isna(fac_next) and pd.isna(exp_next):
                pass
            else:
                assert np.isfinite(exp_next), f"Expected raw value is NaN for next {dt_next}"
                assert np.isclose(float(fac_next), float(exp_next), atol=1e-4, rtol=0.0), (
                    f"Mismatch next {inst} {dt_next.date()}: {factor_name}={fac_next}, raw={exp_next}"
                )
                print(f"Checking next {dt_next.date()}: fac={fac_next}, exp={exp_next}")


def _analyze_raw_profile_factor(csv_path: str, instruments, start: str, end: str, raw_col: str) -> None:
    """Analyze a raw factor from cn_profile.csv before any processing."""
    try:
        prof = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return
    prof.columns = [c.lower() for c in prof.columns]
    if "ann_date" not in prof.columns:
        print("cn_profile.csv missing ann_date column")
        return
    # parse ann_date robustly
    ad_raw = prof["ann_date"]
    try:
        vals = pd.to_numeric(ad_raw, errors="coerce")
        if vals.notna().mean() > 0.5 and (vals.between(19000101, 21001231).mean() > 0.5):
            ad = pd.to_datetime(vals.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            ad = pd.to_datetime(ad_raw.astype(str).str.strip(), errors="coerce")
    except Exception:
        ad = pd.to_datetime(ad_raw, errors="coerce")
    prof["ann_date"] = ad

    # normalize tickers to Qlib format
    def _norm_ticker(x: str) -> str:
        s = str(x).strip().upper()
        if "." in s:
            base, exch = s.split(".")
            base = base[-6:].zfill(6)
            if exch in ("SZ", "SH", "BJ"):
                return f"{exch}{base}"
        return s
    prof["ticker"] = prof.get("ticker", prof.get("code", "")).map(_norm_ticker)

    # Limit to CSI300 instruments
    csi300_list = D.list_instruments(D.instruments(instruments))
    prof = prof[prof["ticker"].isin(csi300_list)]

    # date range filter
    if start:
        prof = prof[prof["ann_date"] >= pd.to_datetime(start)]
    if end:
        prof = prof[prof["ann_date"] <= pd.to_datetime(end)]

    # Check if raw_col exists
    if raw_col not in prof.columns:
        print(f"{raw_col} column not found in cn_profile.csv")
        return

    # numeric conversion with cleaning
    cleaned = (
        prof[raw_col]
        .astype(str)
        .str.strip()
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace('\u2212', '-', regex=False)
        .str.replace('\u2013', '-', regex=False)
        .str.replace('\u2014', '-', regex=False)
    )
    vals = pd.to_numeric(cleaned, errors="coerce")
    finite = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty and finite.quantile(0.95) > 1.5:
        vals = vals / 100.0

    print(f"\nRaw cn_profile {raw_col} distribution:")
    print(vals.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    prof = prof.assign(_val=vals)
    year_q = prof.groupby(prof["ann_date"].dt.year)['_val'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).unstack()
    print(f"\nRaw yearly quantiles ({raw_col}):")
    print(year_q.to_string(float_format=lambda x: f"{x: .6f}"))
    top = prof.loc[vals.abs().nlargest(10).index, ["ticker", "ann_date", raw_col]]
    print(f"\nRaw top absolute {raw_col} values:")
    print(top.to_string(index=False))


def _analyze_factor_distribution(df: pd.DataFrame, fac: str) -> None:
    """Print distribution diagnostics for a factor to spot data issues/outliers."""
    if fac not in df.columns:
        print(f"{fac} not in dataframe; skip distribution analysis.")
        return
    date_level = "datetime" if "datetime" in df.index.names else 0
    s = df[fac].replace([np.inf, -np.inf], np.nan)
    # Overall distribution
    desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"\nOverall distribution of {fac}:")
    print(desc.to_string())
    # Daily sample sizes
    daily_n = s.groupby(level=date_level).apply(lambda x: x.notna().sum())
    print(f"Days with n<30 for {fac}:", int((daily_n < 30).sum()))
    # Yearly quantiles (fundamentals are typically annual/ffill)
    years = df.index.get_level_values(date_level).year
    tmp = pd.DataFrame({fac: s.values}, index=df.index)
    tmp["year"] = years
    year_q = tmp.groupby("year")[fac].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).unstack()
    print(f"\nYearly quantiles for {fac}:")
    print(year_q.to_string(float_format=lambda x: f"{x: .6f}"))
    # Top absolute outliers
    top_idx = s.abs().nlargest(10).index
    top_df = (
        pd.DataFrame({"value": s.loc[top_idx]})
        .reset_index()
        .rename(columns={"level_0": "datetime", "level_1": "instrument"})
    )
    print(f"\nTop absolute outliers for {fac}:")
    print(top_df.to_string(index=False))


def _build_base_index(instruments, start: str, end: str) -> pd.DataFrame:
    """Create a MultiIndex (datetime, instrument) frame with $close, $market_cap and next-day returns."""
    feats = D.features(D.instruments(instruments), ["$close", "$market_cap"], start_time=start, end_time=end)
    feats = feats.sort_index()
    # next-day return per instrument based on $close
    nxt_close = feats[["$close"]].groupby(level="instrument").shift(-1)
    ret1 = (nxt_close - feats[["$close"]]) / feats[["$close"]]
    base = pd.DataFrame(index=feats.index)
    base["$close"] = feats["$close"]
    if "$market_cap" in feats.columns:
        base["$market_cap"] = feats["$market_cap"]
    base["target_ret1"] = ret1["$close"]
    return base


def _apply_fundamental_processors(df: pd.DataFrame) -> pd.DataFrame:
    """Attach F_* and Q_* factors from cn_profile.csv and compute quality factors."""
    df = FundamentalProfileProcessor(csv_path="data/others/cn_profile.csv", prefix="F_")(df)
    df = BarraFundamentalQualityProcessor()(df)
    # Add Industry Momentum based on F_INDUSTRY_CODE and $market_cap/$close
    df = IndustryMomentumProcessor(out_col="B_INDMOM")(df)
    return df


def _compute_ic(df: pd.DataFrame, factors: List[str], target_col: str) -> pd.DataFrame:
    """Compute daily cross-sectional Spearman IC for given factors."""
    results = []
    # Determine the date level name robustly
    date_level = "datetime" if "datetime" in df.index.names else 0
    for fac in factors:
        if fac not in df.columns:
            continue
        ics: List[float] = []
        for dt, g in df[[fac, target_col]].groupby(level=date_level):
            s = g.droplevel(0).replace([np.inf, -np.inf], np.nan).dropna()
            if s.shape[0] < 30:
                print(str(dt), fac, "n=", s.shape[0], "std=", float(s[fac].std(ddof=0)))
                continue
            try:
                # skip if factor is constant that day
                if np.isclose(s[fac].std(ddof=0), 0.0) or np.isclose(s[target_col].std(ddof=0), 0.0):
                    continue
                ic = spearmanr(s[fac], s[target_col]).correlation
            except Exception:
                ic = np.nan
            if np.isfinite(ic):
                ics.append(float(ic))
        if not ics:
            continue
        ics_arr = np.array(ics, dtype=float)
        mean_ic = float(np.mean(ics_arr))
        std_ic = float(np.std(ics_arr, ddof=1)) if len(ics_arr) > 1 else 0.0
        ir = mean_ic / std_ic * np.sqrt(len(ics_arr)) if std_ic > 0 else np.nan
        pos = float((ics_arr > 0).mean())
        results.append((fac, len(ics_arr), mean_ic, std_ic, ir, pos))
    return pd.DataFrame(results, columns=["factor", "n", "mean_ic", "std_ic", "ic_ir", "pos_rate"]).sort_values(
        by="mean_ic", ascending=False
    )


def main():
    qlib.init(provider_uri="data", region="cn")

    # Build universe from cn_profile.csv coverage to avoid all-zero factors
    prof = pd.read_csv("data/others/cn_profile.csv")
    prof.columns = [c.lower() for c in prof.columns]

    # quick sanity: restrict to tickers that have any non-null in key ratio columns
    key_ratio_cols = [
        "debt_to_assets", "return_on_assets", "return_on_equity",
        "asset_turnover", "gross_margin"
    ]
    has_any = prof[key_ratio_cols].notna().any(axis=1)
    prof = prof[has_any]
    def _norm_ticker(x: str) -> str:
        s = str(x).strip().upper()
        if "." in s:
            base, exch = s.split(".")
            base = base[-6:].zfill(6)
            if exch in ("SZ", "SH", "BJ"):
                return f"{exch}{base}"
        return s
    tickers = sorted(set(prof["ticker"].map(_norm_ticker).dropna()))
    # intersect with qlib universe to ensure prices exist
    all_inst = []
    for code in tickers:
        try:
            _ = D.features([code], ["$close"], start_time="2024-01-02", end_time="2024-01-05")
            all_inst.append(code)
        except Exception:
            continue

    # Use instruments with cn_profile coverage and available in provider
    # IMPORTANT: use a static list to avoid dynamic index membership gaps (e.g., SH600004 dropping from CSI300)
    instruments = all_inst if all_inst else "csi300"
    start = "2020-01-01"  # longer window to increase valid samples
    end = "2025-07-01"

    '''
    # Analyze raw for ROE and ROA
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="return_on_equity")
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="return_on_assets")
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="gross_margin")
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="operating_margin")
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="net_margin")
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="debt_to_equity")
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="debt_to_assets")
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="revenue_growth")
    _analyze_raw_profile_factor("data/others/cn_profile.csv", instruments, start, end, raw_col="total_assets_growth")
    '''
    base = _build_base_index(instruments, start, end)
    df = _apply_fundamental_processors(base)

    # Analyze distribution for ROE and ROA
    _analyze_factor_distribution(df, "F_ROE_RATIO")
    _analyze_factor_distribution(df, "F_ROA_RATIO")
    _analyze_factor_distribution(df, "F_GROSS_MARGIN")
    _analyze_factor_distribution(df, "F_OPERATING_MARGIN")
    _analyze_factor_distribution(df, "F_NET_MARGIN")
    _analyze_factor_distribution(df, "F_DTOA_RATIO")
    _analyze_factor_distribution(df, "F_DEBT_TO_EQUITY")
    _analyze_factor_distribution(df, "F_REVENUE_GROWTH")
    _analyze_factor_distribution(df, "F_TOTAL_ASSETS_GROWTH")
    _analyze_factor_distribution(df, "B_INDMOM")

    fac_list = ["F_ROE_RATIO", "F_ROA_RATIO", "F_GROSS_MARGIN", "F_OPERATING_MARGIN", "F_NET_MARGIN", "F_DTOA_RATIO", "F_DEBT_TO_EQUITY", "F_REVENUE_GROWTH", "F_TOTAL_ASSETS_GROWTH", "F_INDUSTRY", "F_INDUSTRY_CODE", "B_INDMOM"]
    # Export F_ROE_RATIO and F_ROA_RATIO to CSV
    for fac in fac_list:
        if fac in df.columns:
            series = df[fac]
            df_out = series.reset_index()  # To flat DataFrame with datetime, instrument, fac
            df_out.to_csv(f"{fac.lower()}.csv", index=False)
            print(f"Exported {fac} to {fac.lower()}.csv")

    # Assert alignment for ROE and ROA
    _assert_factor_alignment(
        df,
        csv_path="data/others/cn_profile.csv",
        raw_ticker="000001.SZ",
        dates=["2024-03-15", "2025-03-15"],
        factor_name="F_ROE_RATIO",
        raw_col="return_on_equity"
    )
    _assert_factor_alignment(
        df,
        csv_path="data/others/cn_profile.csv",
        raw_ticker="000002.SZ",
        dates=["2024-03-15", "2025-03-15"],
        factor_name="F_ROA_RATIO",
        raw_col="return_on_assets"
    )
    _assert_factor_alignment(
        df,
        csv_path="data/others/cn_profile.csv",
        raw_ticker="000002.SZ",
        dates=["2024-03-15", "2025-03-15"],
        factor_name="F_GROSS_MARGIN",
        raw_col="gross_margin"
    )
    _assert_factor_alignment(
        df,
        csv_path="data/others/cn_profile.csv",
        raw_ticker="000002.SZ",
        dates=["2024-03-15", "2025-03-15"],
        factor_name="F_OPERATING_MARGIN",
        raw_col="operating_margin"
    )
    _assert_factor_alignment(
        df,
        csv_path="data/others/cn_profile.csv",
        raw_ticker="000002.SZ",
        dates=["2024-03-15", "2025-03-15"],
        factor_name="F_NET_MARGIN",
        raw_col="net_margin"
    )

    # Debug: check coverage and sample values after processors
    for fac in fac_list:
        if fac in df.columns:
            nn = int(df[fac].notna().sum())
            print(f"{fac} non-null total:", nn)
            by_inst = (
                df[fac].notna().groupby(level="instrument").sum().sort_values(ascending=False)
            )
            print(f"Top instruments by {fac} coverage:\n", by_inst.head(10).to_string())
            sample_non_null = df[df[fac].notna()].head(10)
            if not sample_non_null.empty:
                print(f"Sample non-null {fac} rows:\n", sample_non_null[[fac]].to_string())

    # candidate factor columns
    candidates = [
        # quality (constructed)
        "Q_MLEV", "Q_BLEV", "Q_DTOA", "Q_ATO", "Q_GP", "Q_GPM", "Q_ROA", "Q_ABS", "Q_ACF",
        # direct ratios from cn_profile (if present)
        "F_DTOA_RATIO", "F_DEBT_TO_EQUITY", 
        "F_GROSS_MARGIN", "F_OPERATING_MARGIN", "F_NET_MARGIN", "F_ROA_RATIO", "F_ROE_RATIO",
        # industry momentum
        "B_INDMOM",
    ]
    diag = (
        df[["F_ROE_RATIO", "target_ret1"]]
        .groupby(level=0)
        .apply(lambda g: pd.Series({
            "n": g.dropna().shape[0],
            "std_fac": float(g["F_ROE_RATIO"].replace([np.inf, -np.inf], np.nan).dropna().std(ddof=0) or 0.0),
            "std_tgt": float(g["target_ret1"].replace([np.inf, -np.inf], np.nan).dropna().std(ddof=0) or 0.0),
        }))
    )
    print("days with n<30:", (diag["n"] < 30).sum())
    print("days fac std=0:", (diag["std_fac"] == 0).sum())
    print("days tgt std=0:", (diag["std_tgt"] == 0).sum())
    print(diag.head(10))

    report = _compute_ic(df, candidates, target_col="target_ret1")
    if report.empty:
        print("No valid IC computed - check data availability (cn_profile.csv, prices).")
        return
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    print(report.to_string(index=False, float_format=lambda x: f"{x: .4f}"))


if __name__ == "__main__":
    main()

