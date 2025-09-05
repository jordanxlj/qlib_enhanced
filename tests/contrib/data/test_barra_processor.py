import io
import pandas as pd
import numpy as np

from qlib.contrib.data.barra_processor import (
    FundamentalProfileProcessor,
    BarraFundamentalQualityProcessor,
    IndustryMomentumProcessor,
    BarraFactorProcessor,
    compute_cne6_exposures_ts,
    compute_beta_resvol,
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

    # ME should come from $market_cap when present
    assert "ME" in df1.columns
    assert np.isclose(
        float(df1.loc[(dates[5], instruments[1]), "ME"]),
        float(df.loc[(dates[5], instruments[1]), "$market_cap"]),
    )

    df2 = BarraFundamentalQualityProcessor()(df1.copy())
    # Check several quality factors are present and finite for some rows
    for col in ["Q_MLEV", "Q_DTOA", "Q_ROA", "Q_ABS", "Q_ACF"]:
        assert col in df2.columns
        assert df2[col].notna().any()



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

    out = FundamentalProfileProcessor(csv_path=str(p), prefix="")(df.copy())
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

    out = FundamentalProfileProcessor(csv_path=str(p), prefix="")(df.copy())
    assert "ME" in out.columns
    val_me = out.loc[(dates[-1], instruments[0]), "ME"]
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

    df1 = FundamentalProfileProcessor(csv_path=str(p), prefix="")(df.copy())
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
    df["INDUSTRY_CODE"] = pd.Series(codes)

    proc = IndustryMomentumProcessor(
        mcap_col="$market_cap", window=3, halflife=2, out_col="B_INDMOM"
    )
    out = proc(df.copy())
    assert "B_INDMOM" in out.columns
    # On last date, TECH industry has only one stock, so INDMOM ˜ 0
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
    df["INDUSTRY_CODE"] = pd.Series(codes)

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
    assert "INDUSTRY_CODE" in out.columns and out["INDUSTRY_CODE"].notna().any()

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


def test_compute_cne6_exposures_ts_basic():
    # Build small dataset
    dates = pd.bdate_range("2024-01-01", periods=40)
    instruments = ["SH600000", "SZ000001", "SZ000002"]
    df = _make_multiindex_frame(dates, instruments)
    # Add simple turnover to enable liquidity family
    df["$turnover"] = 1.0

    # Wide pivots
    rets = df["$return"].unstack(level="instrument").sort_index()
    mcap = df["$market_cap"].unstack(level="instrument").sort_index()
    mkt = df["$market_ret"].droplevel("instrument").sort_index()
    turn = df["$turnover"].unstack(level="instrument").sort_index()

    expos_ts = compute_cne6_exposures_ts(
        returns=rets,
        mcap=mcap,
        market_returns=mkt,
        turnover=turn,
        lookbacks={"beta": 10, "resvol": 10, "mom": 10, "mom_gap": 1, "vol_win": 10},
        method_beta="rolling",
        halflife_beta=5,
    )

    # Expected keys
    for key in ["SIZE", "MOM", "BETA", "RESVOL", "VOL", "LIQ", "STOM", "STOQ", "STOA", "ATVR"]:
        assert key in expos_ts
        assert isinstance(expos_ts[key], pd.DataFrame)
        assert expos_ts[key].shape == rets.shape

    # Tails should have valid numbers for at least one instrument after warm-up
    tail_idx = rets.index[-1]
    for key in ["BETA", "RESVOL", "MOM", "VOL", "SIZE", "LIQ", "ATVR"]:
        tail_row = expos_ts[key].loc[tail_idx]
        assert tail_row.notna().any(), f"{key} all NaN at tail"
    # Windows require full lengths; STOM (21), STOQ (63), STOA (252). On 40 days data, STOQ/STOA may be all NaN at tail.


def test_strev_and_season_exposures_ts_long_horizon():
    # Long horizon so SEASON (5y same-month avg) has data
    dates = pd.bdate_range("2010-01-01", "2018-12-31")
    instruments = ["SH600000", "SZ000001", "SH600519"]

    rng = np.random.RandomState(0)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, size=(len(dates), len(instruments))), index=dates, columns=instruments)
    price = (1.0 + rets).cumprod() * 100.0

    rng2 = np.random.RandomState(1)
    mcap = pd.DataFrame(1e9 + 1e8 * rng2.rand(len(dates), len(instruments)), index=dates, columns=instruments)

    rng3 = np.random.RandomState(2)
    mkt = pd.Series(rng3.normal(0.0, 0.008, size=len(dates)), index=dates)

    expos_ts = compute_cne6_exposures_ts(
        returns=rets,
        mcap=mcap,
        market_returns=mkt,
        price=price,
        lookbacks={"mom": 252, "mom_gap": 21, "vol_win": 252, "resvol": 252, "beta": 252},
    )

    assert "STREV" in expos_ts and isinstance(expos_ts["STREV"], pd.DataFrame)
    assert "SEASON" in expos_ts and isinstance(expos_ts["SEASON"], pd.DataFrame)

    strev = expos_ts["STREV"]
    season = expos_ts["SEASON"]

    # Alignment
    assert list(strev.index) == list(dates)
    assert list(season.index) == list(dates)
    assert list(strev.columns) == instruments
    assert list(season.columns) == instruments

    # STREV warm-up (first 20 days NaN), later some values
    assert strev.iloc[:20].isna().all().all()
    assert strev.iloc[30:].notna().any().any()

    # SEASON should have some non-null in the last year
    last_year = season.loc[season.index >= (season.index.max() - pd.Timedelta(days=365))]
    assert last_year.notna().any().any()


def test_strev_exactness_small_series():
    # Constant returns: STREV after warm-up equals the constant (weights sum to 1)
    dates = pd.bdate_range("2024-01-01", periods=40)
    instruments = ["SH600000"]
    r = 0.01
    rets = pd.DataFrame(r, index=dates, columns=instruments)
    mcap = pd.DataFrame(1e9, index=dates, columns=instruments)
    mkt = pd.Series(0.0, index=dates)

    expos_ts = compute_cne6_exposures_ts(
        returns=rets,
        mcap=mcap,
        market_returns=mkt,
    )
    strev = expos_ts["STREV"][instruments[0]]
    # After 21-day warm-up, STREV should equal ~0.01
    tail_val = float(strev.iloc[-1])
    assert np.isfinite(tail_val)
    assert abs(tail_val - r) < 1e-9


def test_momentum_exactness_with_gap():
    # Constant returns -> MOM equals log1p(r) after warm-up and shifted by gap
    dates = pd.bdate_range("2024-01-01", periods=30)
    instruments = ["SH600000"]
    r = 0.005
    rets = pd.DataFrame(r, index=dates, columns=instruments)
    mcap = pd.DataFrame(1e9, index=dates, columns=instruments)
    mkt = pd.Series(0.0, index=dates)

    expos_ts = compute_cne6_exposures_ts(
        returns=rets,
        mcap=mcap,
        market_returns=mkt,
        lookbacks={"mom": 5, "mom_gap": 2, "halflife_mom": 2},
    )
    mom = expos_ts["MOM"][instruments[0]]
    # pick a position far enough after warm-up+gap
    expected = np.log1p(r)
    assert abs(float(mom.iloc[-1]) - expected) < 1e-9


def test_beta_resvol_identity_market():
    # Each stock identical to market -> beta˜1, resvol˜0 on tail
    dates = pd.bdate_range("2024-01-01", periods=40)
    instruments = ["SH600000", "SZ000001"]
    mkt = pd.Series(np.linspace(-0.01, 0.02, len(dates)), index=dates)
    rets = pd.DataFrame({inst: mkt.values for inst in instruments}, index=dates)
    mcap = pd.DataFrame(1e9, index=dates, columns=instruments)

    expos_ts = compute_cne6_exposures_ts(
        returns=rets,
        mcap=mcap,
        market_returns=mkt,
        lookbacks={"beta": 10, "resvol": 10, "mom": 5, "mom_gap": 1, "vol_win": 10},
        method_beta="rolling",
        halflife_beta=5,
    )
    beta_tail = expos_ts["BETA"].iloc[-1]
    resvol_tail = expos_ts["RESVOL"].iloc[-1]
    assert np.allclose(beta_tail.values, 1.0, atol=1e-6, rtol=0)
    assert np.allclose(resvol_tail.values, 0.0, atol=1e-10, rtol=0)


def test_size_from_ts_snapshot():
    # SIZE equals -log(ME) snapshot on tail
    dates = pd.bdate_range("2024-01-01", periods=10)
    instruments = ["SH600000", "SZ000001"]
    rets = pd.DataFrame(0.0, index=dates, columns=instruments)
    mcap = pd.DataFrame([[1e10, 1e12]] * len(dates), index=dates, columns=instruments)
    mkt = pd.Series(0.0, index=dates)
    expos_ts = compute_cne6_exposures_ts(rets, mcap, mkt)
    size_tail = expos_ts["SIZE"].iloc[-1]
    expected = -np.log(mcap.iloc[-1])
    assert np.allclose(size_tail.values, expected.values)


def test_momentum_constant_returns_long_window():
    dates = pd.bdate_range("2024-01-01", periods=600)
    instruments = ["SH600000"]
    r = 0.005
    rets = pd.DataFrame(r, index=dates, columns=instruments)
    mcap = pd.DataFrame(1e9, index=dates, columns=instruments)
    mkt = pd.Series(0.0, index=dates)
    expos_ts = compute_cne6_exposures_ts(
        returns=rets,
        mcap=mcap,
        market_returns=mkt,
        lookbacks={"mom": 504, "mom_gap": 21, "halflife_mom": 126},
    )
    mom_tail = float(expos_ts["MOM"][instruments[0]].iloc[-1])
    assert abs(mom_tail - np.log1p(r)) < 1e-9


def test_beta_resvol_slope_approx():
    # Stock ~ 1.5 * market + noise -> beta ~ 1.5
    rng = np.random.RandomState(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    mkt = pd.Series(rng.normal(0.0, 0.02, len(dates)), index=dates)
    rets = pd.DataFrame(1.5 * mkt.values + rng.normal(0.0, 0.01, len(dates)), index=dates, columns=["SH600000"])
    beta, resvol = compute_beta_resvol(rets, mkt, beta_win=252, resvol_win=252, halflife_beta=63)
    assert np.isclose(float(beta.iloc[0]), 1.5, atol=0.1)
    assert 0.0 < float(resvol.iloc[0]) < 0.05


def test_volatility_monotonicity():
    # Higher variance returns should produce higher VOL tail (all else equal)
    dates = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.RandomState(42)
    mkt = pd.Series(rng.normal(0.0, 0.01, len(dates)), index=dates)
    low = pd.DataFrame(np.random.normal(0.0, 0.01, len(dates)), index=dates, columns=["SH600000"])
    high = pd.DataFrame(np.random.normal(0.0, 0.03, len(dates)), index=dates, columns=["SH600000"])
    mcap = pd.DataFrame(1e9, index=dates, columns=["SH600000"])
    vol_low = compute_cne6_exposures_ts(low, mcap, mkt, lookbacks={"vol_win": 252})["VOL"].iloc[-1, 0]
    vol_high = compute_cne6_exposures_ts(high, mcap, mkt, lookbacks={"vol_win": 252})["VOL"].iloc[-1, 0]
    assert vol_high > vol_low


def test_liquidity_constant_rate_ts():
    dates = pd.bdate_range("2024-01-01", periods=300)
    instruments = ["SH600000"]
    rets = pd.DataFrame(0.0, index=dates, columns=instruments)
    mcap = pd.DataFrame(1e9, index=dates, columns=instruments)
    mkt = pd.Series(0.0, index=dates)
    turn = pd.DataFrame(0.02, index=dates, columns=instruments)
    expos_ts = compute_cne6_exposures_ts(rets, mcap, mkt, turnover=turn)
    liq_tail = float(expos_ts["LIQ"].iloc[-1, 0])
    expected = np.log1p(0.02)
    assert abs(liq_tail - expected) < 1e-9


def test_earnings_yield_ts():
    dates = pd.bdate_range("2024-01-01", periods=10)
    instruments = ["SH600000"]
    price = pd.DataFrame(20.0, index=dates, columns=instruments)
    eps_tr = pd.DataFrame(2.0, index=dates, columns=instruments)
    eps_pred = pd.DataFrame(2.5, index=dates, columns=instruments)
    rets = pd.DataFrame(0.0, index=dates, columns=instruments)
    mcap = pd.DataFrame(1e9, index=dates, columns=instruments)
    mkt = pd.Series(0.0, index=dates)
    out = compute_cne6_exposures_ts(rets, mcap, mkt, price=price, eps_trailing=eps_tr, eps_pred=eps_pred)
    ey_tail = float(out["EY"].iloc[-1, 0])
    expected = 0.5 * (2.0 / 20.0) + 0.5 * (2.5 / 20.0)
    assert abs(ey_tail - expected) < 1e-12


def test_industry_momentum_constant_same_industry():
    dates = pd.bdate_range("2024-01-01", periods=200)
    instruments = ["SH600000", "SZ000001"]
    df = _make_multiindex_frame(dates, instruments)
    df["$market_cap"] = 1.0e8
    codes = {(d, instruments[0]): "IND1" for d in dates}
    codes.update({(d, instruments[1]): "IND1" for d in dates})
    df["INDUSTRY_CODE"] = pd.Series(codes)
    out = IndustryMomentumProcessor(window=126, halflife=21, out_col="B_INDMOM")(df.copy())
    last = out["B_INDMOM"].groupby(level="instrument").last()
    assert np.allclose(last.values, 0.0, atol=1e-3)