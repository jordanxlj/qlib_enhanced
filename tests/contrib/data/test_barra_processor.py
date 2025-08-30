import io
import pandas as pd
import numpy as np

from qlib.contrib.data.barra_processor import (
    FundamentalProfileProcessor,
    BarraFundamentalQualityProcessor,
    BarraCNE6Processor,
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
    df["$mkt_cap"] = (ser_close * 1e8).astype(float)
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
    sub = df1.loc[(dates[3], instruments[0]), "F_ROE_RATIO"]
    assert pd.notna(sub)

    # F_ME should come from $mkt_cap when present
    assert "F_ME" in df1.columns
    assert np.isclose(
        float(df1.loc[(dates[5], instruments[1]), "F_ME"]),
        float(df.loc[(dates[5], instruments[1]), "$mkt_cap"]),
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
    assert np.isclose(float(vals), -0.05)


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
    assert np.isclose(float(out.loc[(dates[-1], instruments[0]), "F_ROE_RATIO"]), 0.20)


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
    # remove $mkt_cap to force fallback
    df = df.drop(columns=["$mkt_cap"]) 

    rows = [
        dict(ticker="600000.SH", period="annual", ann_date=20240103, market_cap=1.23e11),
    ]
    p = tmp_path / "cn_profile.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    out = FundamentalProfileProcessor(csv_path=str(p), prefix="F_")(df.copy())
    assert "F_ME" in out.columns
    assert np.isclose(float(out.loc[(dates[-1], instruments[0]), "F_ME"]), 1.23e11)


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


