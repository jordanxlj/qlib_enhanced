import io
import pandas as pd
import numpy as np

from qlib.contrib.data.barra_processor import (
    FundamentalProfileProcessor,
    BarraFundamentalQualityProcessor,
    BarraCNE6Processor,
    IndustryMomentumProcessor,
)


def _make_multiindex_frame(dates, instruments) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    df = pd.DataFrame(index=idx)
    # deterministic values for reproducibility
    base = np.linspace(10.0, 12.0, num=len(dates))
    close = []
    for i, _ in enumerate(instruments):
        close.append(base + i * 0.1)
    close = np.vstack(close).T  # shape: len(dates) x len(instruments)
    # flatten into MultiIndex series
    ser_close = pd.Series(
        close.reshape(-1), index=idx, dtype=float
    )
    df["$close"] = ser_close
    # simple market cap scaled by close
    df["$market_cap"] = (ser_close * 1e8).astype(float)
    # explicit return per instrument
    rets = df["$close"].groupby(level="instrument").pct_change().fillna(0.0)
    df["$return"] = rets
    # market return as cross-sectional mean of returns per day
    mkt_by_date = rets.groupby(level="datetime").mean()
    df["$market_ret"] = mkt_by_date.loc[df.index.get_level_values("datetime")].values
    return df


def _make_cn_profile_csv(tmp_path, instruments):
    # build a tiny cn_profile CSV with two announcement dates
    rows = []
    for inst in instruments:
        # convert to provider code like 600000.SH
        exch, code = inst[:2], inst[2:]
        code_dot = f"{code}.{exch}"
        # two yearly records
        rows.append(
            dict(
                ticker=code_dot,
                period="annual",
                ann_date=20240103,
                market_cap=1.5e11,
                return_on_equity=0.12,
                shareholders_equity=1.5e11,
                total_assets=5.0e11,
                net_income=1.8e10,
                total_liabilities=3.5e11,
                revenue=8.0e10,
                gross_profit=3.2e10,
                operating_cash_flow=1.0e10,
                capital_expenditure=-2.0e9,
                depreciation_and_amortization=1.5e9,
            )
        )
        rows.append(
            dict(
                ticker=code_dot,
                period="annual",
                ann_date=20240108,
                market_cap=1.6e11,
                return_on_equity=0.15,
                shareholders_equity=1.6e11,
                total_assets=5.1e11,
                net_income=2.0e10,
                total_liabilities=3.5e11,
                revenue=8.5e10,
                gross_profit=3.4e10,
                operating_cash_flow=1.2e10,
                capital_expenditure=-2.2e9,
                depreciation_and_amortization=1.6e9,
            )
        )
    df = pd.DataFrame(rows)
    p = tmp_path / "cn_profile.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_fundamental_profile_and_quality(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=10)
    instruments = ["SH600000", "SZ000001"]
    df = _make_multiindex_frame(dates, instruments)

    csv_path = _make_cn_profile_csv(tmp_path, instruments)

    # Apply processors
    df1 = FundamentalProfileProcessor(csv_path=csv_path, prefix="F_")(df.copy())
    assert "F_ROE_RATIO" in df1.columns
    # values should be forward-filled across dates within each instrument
    # First non-null should appear at/after first announcement date 2024-01-03
    val_a = df1.loc[(pd.Timestamp("2024-01-03"), instruments[0]), "F_ROE_RATIO"]
    val_b = df1.loc[(pd.Timestamp("2024-01-04"), instruments[0]), "F_ROE_RATIO"]
    assert (not pd.isna(val_a)) or (not pd.isna(val_b))

    # F_ME should come from $market_cap when present
    assert "F_ME" in df1.columns
    assert np.isclose(
        float(df1.loc[(dates[5], instruments[1]), "F_ME"]),
        float(df.loc[(dates[5], instruments[1]), "$market_cap"]),
    )

    df2 = BarraFundamentalQualityProcessor()(df1.copy())
    # Check several quality factors are present and finite for some rows
    for col in ["Q_MLEV", "Q_DTOA", "Q_ROA", "Q_ABS", "Q_ACF"]:
        assert col in df2.columns
        assert df2[col].notna().any()


def test_barra_cne6_exposures_small_window():
    dates = pd.bdate_range("2024-01-01", periods=30)
    instruments = ["SH600000", "SZ000001"]
    df = _make_multiindex_frame(dates, instruments)

    proc = BarraCNE6Processor(market_col="$market_ret", lookbacks={"beta": 5, "resvol": 5, "mom": 5, "mom_gap": 1, "vol_win": 5})
    out = proc(df.copy())
    # Ensure exposure columns exist and have values
    for col in ["B_SIZE", "B_BETA", "B_MOM", "B_RESVOL", "B_VOL"]:
        assert col in out.columns
        assert out[col].notna().any()


def test_exposures_when_only_close_present():
    # remove $return to force processor to derive returns from $close
    dates = pd.bdate_range("2024-01-01", periods=40)
    instruments = ["SH600000", "SZ000001"]
    df = _make_multiindex_frame(dates, instruments)
    df = df.drop(columns=["$return"])  # only $close, $market_cap, $market_ret remain

    proc = BarraCNE6Processor(market_col="$market_ret", lookbacks={"beta": 5, "resvol": 5, "mom": 5, "mom_gap": 1, "vol_win": 5})
    out = proc(df.copy())
    for col in ["B_SIZE", "B_BETA", "B_MOM", "B_RESVOL", "B_VOL"]:
        assert col in out.columns
        assert out[col].notna().any()


def test_missing_roe_column(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=5)
    instruments = ["SH600000"]
    df = _make_multiindex_frame(dates, instruments)

    # CSV without return_on_equity
    rows = [
        dict(
            ticker="600000.SH",
            period="annual",
            ann_date=20240103,
            shareholders_equity=1.0e11,
        )
    ]
    p = tmp_path / "cn_profile.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    out = FundamentalProfileProcessor(csv_path=str(p), prefix="F_")(df.copy())
    assert "F_ROE_RATIO" not in out.columns or out["F_ROE_RATIO"].isna().all()


def test_percentage_and_unicode_minus_parsing(tmp_path):
    # include dates beyond the second announcement (2024-01-08)
    dates = pd.bdate_range("2024-01-01", periods=10)
    instruments = ["SH600000"]
    df = _make_multiindex_frame(dates, instruments)

    rows = [
        dict(ticker="600000.SH", period="annual", ann_date=20240103, return_on_equity="12.3%"),
        dict(ticker="600000.SH", period="annual", ann_date=20240108, return_on_equity="\u22125.0%"),
    ]
    p = tmp_path / "cn_profile.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    out = FundamentalProfileProcessor(csv_path=str(p), prefix="F_")(df.copy())
    # After 2024-01-08, value should be -0.05, forward-filled across dates
    # pick a date well after the second announcement
    vals = out.loc[(pd.Timestamp("2024-01-09"), instruments[0]), "F_ROE_RATIO"]
    assert not pd.isna(vals) and np.isclose(float(vals), -0.05)


def test_duplicate_ann_date_last_wins(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=4)
    instruments = ["SH600000"]
    df = _make_multiindex_frame(dates, instruments)

    rows = [
        dict(ticker="600000.SH", period="annual", ann_date=20240103, return_on_equity=0.10),
        dict(ticker="600000.SH", period="annual", ann_date=20240103, return_on_equity=0.20),  # duplicate, keep last
    ]
    p = tmp_path / "cn_profile.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    out = FundamentalProfileProcessor(csv_path=str(p), prefix="F_")(df.copy())
    # Any date on/after 2024-01-03 should reflect 0.20
    val_dup = out.loc[(dates[-1], instruments[0]), "F_ROE_RATIO"]
    assert not pd.isna(val_dup) and np.isclose(float(val_dup), 0.20)


def test_ignore_tickers_not_in_df(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=3)
    instruments = ["SH600000"]
    df = _make_multiindex_frame(dates, instruments)

    rows = [
        dict(ticker="000001.SZ", period="annual", ann_date=20240103, return_on_equity=0.10),
    ]
    p = tmp_path / "cn_profile.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    out = FundamentalProfileProcessor(csv_path=str(p), prefix="F_")(df.copy())
    # No values should be assigned since csv ticker not in df instruments
    assert "F_ROE_RATIO" not in out.columns or out["F_ROE_RATIO"].isna().all()


def test_f_me_fallback_to_profile_market_cap(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=3)
    instruments = ["SH600000"]
    df = _make_multiindex_frame(dates, instruments)
    # remove $market_cap to force fallback
    df = df.drop(columns=["$market_cap"])

    rows = [
        dict(ticker="600000.SH", period="annual", ann_date=20240103, market_cap=1.23e11),
    ]
    p = tmp_path / "cn_profile.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    out = FundamentalProfileProcessor(csv_path=str(p), prefix="F_")(df.copy())
    assert "F_ME" in out.columns
    val_me = out.loc[(dates[-1], instruments[0]), "F_ME"]
    assert not pd.isna(val_me) and np.isclose(float(val_me), 1.23e11)


def test_quality_nan_preserved(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=3)
    instruments = ["SH600000"]
    df = _make_multiindex_frame(dates, instruments)

    rows = [
        dict(
            ticker="600000.SH", period="annual", ann_date=20240103,
            total_liabilities=3.0, total_assets=np.nan
        ),
    ]
    p = tmp_path / "cn_profile.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    df1 = FundamentalProfileProcessor(csv_path=str(p), prefix="F_")(df.copy())
    df2 = BarraFundamentalQualityProcessor()(df1)
    # Q_DTOA should be NaN when TA is NaN (no zero-fill)
    assert df2["Q_DTOA"].isna().any()


def test_industry_momentum_basic():
    dates = pd.bdate_range("2024-01-01", periods=8)
    instruments = ["SH600000", "SZ000001", "SZ000002"]
    df = _make_multiindex_frame(dates, instruments)
    # Make equal caps for stability and equal weights in BANK industry
    df["$market_cap"] = 1.0e8
    # Attach industry codes: two in BANK, one in TECH
    codes = {
        (d, instruments[0]): "BANK" for d in dates
    }
    codes.update({(d, instruments[1]): "BANK" for d in dates})
    codes.update({(d, instruments[2]): "TECH" for d in dates})
    df["F_INDUSTRY_CODE"] = pd.Series(codes)

    proc = IndustryMomentumProcessor(
        mcap_col="$market_cap", window=3, halflife=2, out_col="B_INDMOM"
    )
    out = proc(df.copy())
    assert "B_INDMOM" in out.columns
    # On last date, TECH industry has only one stock, so INDMOM ¡Ö 0
    d_last = dates[-1]
    tech_val = float(out.loc[(d_last, instruments[2]), "B_INDMOM"])
    assert np.isfinite(tech_val) and np.isclose(tech_val, 0.0, atol=1e-10)

    # For BANK industry with equal weights 0.5, verify
    # INDMOM1 - INDMOM2 = -0.5*(RS1 - RS2)
    # Recompute RS using same definition
    rets = out["$return"].unstack(level="instrument").sort_index()
    weights = np.exp(-np.log(2) * np.arange(3) / 2)[::-1]
    weights /= weights.sum()
    log1p = np.log1p(rets.fillna(0.0))
    rs_df = log1p.rolling(3).apply(lambda x: np.nansum(weights * np.nan_to_num(x)), raw=True)
    rs1 = float(rs_df[instruments[0]].iloc[-1])
    rs2 = float(rs_df[instruments[1]].iloc[-1])
    indmom1 = float(out.loc[(d_last, instruments[0]), "B_INDMOM"])
    indmom2 = float(out.loc[(d_last, instruments[1]), "B_INDMOM"])
    assert np.isfinite(indmom1) and np.isfinite(indmom2)
    assert np.isclose(indmom1 - indmom2, -0.5 * (rs1 - rs2), atol=1e-10)


def test_industry_momentum_index_column_name_robustness():
    dates = pd.bdate_range("2024-01-01", periods=12)
    instruments = ["SH600000", "SZ000001", "SZ000002"]
    df = _make_multiindex_frame(dates, instruments)
    df["$market_cap"] = 1.0e8
    codes = {(d, instruments[0]): "BANK" for d in dates}
    codes.update({(d, instruments[1]): "BANK" for d in dates})
    codes.update({(d, instruments[2]): "TECH" for d in dates})
    df["F_INDUSTRY_CODE"] = pd.Series(codes)

    proc = IndustryMomentumProcessor(mcap_col="$market_cap", window=4, halflife=2, out_col="B_INDMOM")

    # Case 1: names present
    out1 = proc(df.copy())
    assert out1["B_INDMOM"].notna().any()

    # Case 2: drop index names
    df2 = df.copy()
    df2.index = df2.index.set_names([None, None])
    out2 = proc(df2)
    assert out2["B_INDMOM"].notna().any()

    # Case 3: swap level order (instrument, datetime)
    df3 = df.copy()
    df3.index = df3.index.reorder_levels(["instrument", "datetime"])  # reorder only
    df3 = df3.sort_index()
    out3 = proc(df3)
    if not out3["B_INDMOM"].notna().any():
        print("All NaN in swapped case; non-null:", out3["B_INDMOM"].notna().sum())
    assert out3["B_INDMOM"].notna().any()


def _make_facts_csv(tmp_path, instruments):
    # instruments like SH600000 -> 600000.SH
    rows = []
    for inst in instruments:
        exch, code = inst[:2], inst[2:]
        rows.append({
            "ticker": f"{code}.{exch}",
            "industry_code": "BANK" if inst != instruments[-1] else "TECH",
            "industry": "Bank" if inst != instruments[-1] else "Tech",
        })
    p = tmp_path / "cn_facts.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_barra_factor_processor_basic(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=12)
    instruments = ["SH600000", "SZ000001", "SZ000002"]
    df = _make_multiindex_frame(dates, instruments)
    # Ensure market cap present and stable
    df["$market_cap"] = (df["$close"] * 1e8).astype(float)

    profile_csv = _make_cn_profile_csv(tmp_path, instruments)
    facts_csv = _make_facts_csv(tmp_path, instruments)

    proc = BarraFactorProcessor(
        profile_csv_path=profile_csv,
        facts_csv_path=facts_csv,
        market_col="$market_ret",
        price_col="$close",
        lookbacks={"beta": 5, "resvol": 5, "mom": 5, "mom_gap": 1, "vol_win": 5},
        indmom_window=4,
        indmom_halflife=2,
        debug=False,
    )
    out = proc(df.copy())

    # Check fundamentals / quality / industry
    assert "F_ROE_RATIO" in out.columns or "F_RETURN_ON_EQUITY" in out.columns
    for col in ["Q_MLEV", "Q_DTOA", "Q_ROA"]:
        assert col in out.columns and out[col].notna().any()
    assert "F_INDUSTRY_CODE" in out.columns and out["F_INDUSTRY_CODE"].notna().any()

    # Check CNE6 exposures
    for col in ["B_SIZE", "B_BETA", "B_MOM", "B_RESVOL"]:
        assert col in out.columns and out[col].notna().any()

    # Check industry momentum
    assert "B_INDMOM" in out.columns and out["B_INDMOM"].notna().any()


def test_barra_factor_processor_index_column_name_robustness(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=12)
    instruments = ["SH600000", "SZ000001", "SZ000002"]
    df = _make_multiindex_frame(dates, instruments)
    df["$market_cap"] = (df["$close"] * 1e8).astype(float)

    profile_csv = _make_cn_profile_csv(tmp_path, instruments)
    facts_csv = _make_facts_csv(tmp_path, instruments)

    proc = BarraFactorProcessor(
        profile_csv_path=profile_csv,
        facts_csv_path=facts_csv,
        market_col="$market_ret",
        price_col="$close",
        lookbacks={"beta": 5, "resvol": 5, "mom": 5, "mom_gap": 1, "vol_win": 5},
        indmom_window=4,
        indmom_halflife=2,
        debug=False,
    )

    # Case A: drop index names
    df2 = df.copy()
    df2.index = df2.index.set_names([None, None])
    out2 = proc(df2)
    assert out2["B_INDMOM"].notna().any()

    # Case B: swap levels
    df3 = df.copy()
    df3.index = df3.index.reorder_levels(["instrument", "datetime"])
    df3 = df3.sort_index()
    out3 = proc(df3)
    assert out3["B_INDMOM"].notna().any()