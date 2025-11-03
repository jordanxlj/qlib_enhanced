from qlib.data.dataset.loader import QlibDataLoader


class Alpha360DL(QlibDataLoader):
    """Dataloader to get Alpha360"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        # NOTE:
        # Alpha360 tries to provide a dataset with original price data
        # the original price data includes the prices and volume in the last 60 days.
        # To make it easier to learn models from this dataset, all the prices and volume
        # are normalized by the latest price and volume data ( dividing by $close, $volume)
        # So the latest normalized $close will be 1 (with name CLOSE0), the latest normalized $volume will be 1 (with name VOLUME0)
        # If further normalization are executed (e.g. centralization),  CLOSE0 and VOLUME0 will be 0.
        fields = []
        names = []

        for i in range(59, 0, -1):
            fields += ["Ref($close, %d)/$close" % i]
            names += ["CLOSE%d" % i]
        fields += ["$close/$close"]
        names += ["CLOSE0"]
        for i in range(59, 0, -1):
            fields += ["Ref($open, %d)/$close" % i]
            names += ["OPEN%d" % i]
        fields += ["$open/$close"]
        names += ["OPEN0"]
        for i in range(59, 0, -1):
            fields += ["Ref($high, %d)/$close" % i]
            names += ["HIGH%d" % i]
        fields += ["$high/$close"]
        names += ["HIGH0"]
        for i in range(59, 0, -1):
            fields += ["Ref($low, %d)/$close" % i]
            names += ["LOW%d" % i]
        fields += ["$low/$close"]
        names += ["LOW0"]
        for i in range(59, 0, -1):
            fields += ["Ref($vwap, %d)/$close" % i]
            names += ["VWAP%d" % i]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]
        for i in range(59, 0, -1):
            fields += ["Ref($volume, %d)/($volume+1e-12)" % i]
            names += ["VOLUME%d" % i]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]

        return fields, names


class Alpha158DL(QlibDataLoader):
    """Dataloader to get Alpha158"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config(
        config={
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
    ):
        """create factors from config

        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            }
        }
        """
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field

            def use(x):
                return x not in exclude and (include is None or x in include)

            # Some factor ref: https://guorn.com/static/upload/file/3/134065454575605.pdf
            if use("ROC"):
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # Rate of change, the price change in the past d days, divided by latest close price to remove unit
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("STD"):
                # The standard diviation of close price for the past d days, divided by latest close price to remove unit
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                # The rate of close price change in the past d days, divided by latest close price to remove unit
                # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                # The R-sqaure value of linear regression for the past d days, represent the trend linear
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                # The max price for past d days, divided by latest close price to remove unit
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                # The low price for past d days, divided by latest close price to remove unit
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                # The 80% quantile of past d day's close price, divided by latest close price to remove unit
                # Used with MIN and MAX
                fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("QTLD"):
                # The 20% quantile of past d day's close price, divided by latest close price to remove unit
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                # Get the percentile of current close price in past d day's close price.
                # Represent the current price level comparing to past N days, add additional information to moving average.
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RSV"):
                # Represent the price position between upper and lower resistent price for past d days.
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                # The number of days between current date and previous highest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                # The number of days between current date and previous lowest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                # The time period between previous lowest-price date occur after highest price date.
                # Large value suggest downward momemtum.
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                # The correlation between absolute close price and log scaled trading volume
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                # The correlation between price change ratio and volume change ratio
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                # The percentage of days in past d days that price go up.
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                # The percentage of days in past d days that price go down.
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("CNTD"):
                # The diff between past up day and past down day
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                # The total gain / the absolute total price changed
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMN"):
                # The total lose / the absolute total price changed
                # Can be derived from SUMP by SUMN = 1 - SUMP
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                # The diff ratio between total gain and total lose
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                # The standard deviation for volume in past d days.
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                # The volume weighted price change volatility
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                # The total volume increase / the absolute total volume changed
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                # The total volume increase / the absolute total volume changed
                # Can be derived from VSUMP by VSUMN = 1 - VSUMP
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                # The diff ratio between total volume increase and total volume decrease
                # RSI indicator for volume
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]

        return fields, names


class AlphaCNE6DL(Alpha158DL):
    """Dataloader to get Alpha158 base features plus raw series for CNE6.

    Extends Alpha158DL by adding the following raw columns for the processor pipeline:
      - $mkt_cap: read from market_cap.day.bin
      - $turnover: read from turnover.day.bin
      - $dividend_ratio: read from dividend_ratio.day.bin

    Notes
    -----
    - City-value reference uses $close alignment as in Alpha158.
    - Downstream processors (e.g., BarraCNE6Processor) will consume these columns
      to compute Barra/CNE6 exposures.
    """

    @staticmethod
    def get_feature_config(base_config=None):
        # Start from Alpha158 default config
        fields, names = Alpha158DL.get_feature_config(config=base_config or {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP", "CLOSE"]},
            "rolling": {},
        })

        # Append raw series needed by CNE6 processors
        # These field names should map to provider columns backed by the corresponding *.day.bin files.
        extra_fields = [
            "$mkt_cap",          # market_cap.day.bin
            "$turnover",         # turnover.day.bin
            "$dividend_ratio",   # dividend_ratio.day.bin
        ]
        extra_names = [
            "MKT_CAP",
            "TURNOVER",
            "DIV_RATIO",
        ]
        fields += extra_fields
        names += extra_names

        # Optionally include a simple return column for convenience
        # Downstream processors can derive if missing; keep here for completeness.
        if "$return" not in fields:
            fields += ["$close/Ref($close,1)-1"]
            names += ["RETURN1"]

        # Liquidity factors (CNE6-style approximations):
        # STOM: ln(sum of daily turnover over last 21 trading days)
        # STOQ: ln(average of 3 monthly STOMs) ¡Ö ln(mean of 21-day turnover sums over 63 days)
        # STOA: ln(average of 12 monthly STOMs) ¡Ö ln(mean of 21-day turnover sums over 252 days)
        # ATR:  annualized traded value ratio (approx) = mean turnover over 252 days
        liq_fields = [
            "Log(Sum($turnover, 21)+1e-12)",            # LIQ_STOM
            "Log(Mean(Sum($turnover, 21), 63)+1e-12)",  # LIQ_STOQ
            "Log(Mean(Sum($turnover, 21), 252)+1e-12)", # LIQ_STOA
            "Mean($turnover, 252)",                      # LIQ_ATR (approx, no decay)
        ]
        liq_names = [
            "LIQ_STOM",
            "LIQ_STOQ",
            "LIQ_STOA",
            "LIQ_ATR",
        ]
        fields += liq_fields
        names += liq_names

        # Size factors
        # LNCAP: log of free-float market capitalization
        # MIDCAP (approx raw): cubic of LNCAP; orthogonalization/standardization can be handled by processors
        size_fields = [
            "Log($mkt_cap+1e-12)",
            "(Log($mkt_cap+1e-12)*Log($mkt_cap+1e-12)*Log($mkt_cap+1e-12))",
        ]
        size_names = [
            "LNCAP",
            "MIDCAP_RAW",
        ]
        fields += size_fields
        names += size_names

        return fields, names


class EnhancedAlpha158DL(Alpha158DL):
    """Dataloader to get Alpha158 base features plus additional fundamental and technical factors.
    
    Extends Alpha158DL by adding enhanced factors with industry neutralization.
    All additional factors are neutralized by industry using IndustryNeutralizeProcessor.
    """

    @staticmethod
    def get_feature_config(base_config=None):
        # Start from Alpha158 default config
        fields, names = Alpha158DL.get_feature_config(config=base_config or {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP", "CLOSE"]},
            "rolling": {},
        })

        # Additional factors to add (industry neutralized)
        # Format: factor_name -> field_name (assuming these map to $field_name in provider)
        additional_factors = [
            ("ar_turn", "$ar_turn"),
            ("assets_turn", "$assets_turn"),
            ("assets_yoy", "$assets_yoy"),
            ("basic_eps_yoy", "$basic_eps_yoy"),
            ("bps", "$bps"),
            ("ca_turn", "$ca_turn"),
            ("cash_ratio", "$cash_ratio"),
            ("cfps", "$cfps"),
            ("cost_15pct", "$cost_15pct"),
            ("cost_50pct", "$cost_50pct"),
            ("cost_5pct", "$cost_5pct"),
            ("cost_85pct", "$cost_85pct"),
            ("cost_95pct", "$cost_95pct"),
            ("current_ratio", "$current_ratio"),
            ("debt_to_assets", "$debt_to_assets"),
            ("debt_to_ebitda", "$debt_to_ebitda"),
            ("debt_to_eqt", "$debt_to_eqt"),
            ("dv_ratio", "$dv_ratio"),
            ("eps_ttm", "$eps_ttm"),
            ("equity_yoy", "$equity_yoy"),
            ("f_dv_ratio", "$f_dv_ratio"),
            ("f_eps", "$f_eps"),
            ("f_neg_ratio", "$f_neg_ratio"),
            ("f_pe", "$f_pe"),
            ("f_pos_ratio", "$f_pos_ratio"),
            ("f_roe", "$f_roe"),
            ("f_target_price", "$f_target_price"),
            ("fa_turn", "$fa_turn"),
            ("fcf_margin_ttm", "$fcf_margin_ttm"),
            ("fcff_ps", "$fcff_ps"),
            ("goodwill", "$goodwill"),
            ("grossprofit_margin_ttm", "$grossprofit_margin_ttm"),
            ("inv_turn", "$inv_turn"),
            ("main_inflow_ratio", "$main_inflow_ratio"),
            ("market_cap", "$market_cap"),
            ("net_inflow_ratio", "$net_inflow_ratio"),
            ("netincome_cagr_3y", "$netincome_cagr_3y"),
            ("netprofit_margin_ttm", "$netprofit_margin_ttm"),
            ("netprofit_yoy", "$netprofit_yoy"),
            ("ocf_yoy", "$ocf_yoy"),
            ("open", "$open"),
            ("or_yoy", "$or_yoy"),
            ("pb", "$pb"),
            ("pe", "$pe"),
            ("ps", "$ps"),
            ("quick_ratio", "$quick_ratio"),
            ("rd_exp_to_capex", "$rd_exp_to_capex"),
            ("revenue_cagr_3y", "$revenue_cagr_3y"),
            ("revenue_ps_ttm", "$revenue_ps_ttm"),
            ("roa_ttm", "$roa_ttm"),
            ("roe_ttm", "$roe_ttm"),
            ("roe_yoy", "$roe_yoy"),
            ("roic", "$roic"),
            ("small_inflow_ratio", "$small_inflow_ratio"),
            ("turnover", "$turnover"),
            ("volume_ratio", "$volume_ratio"),
            ("weight_avg", "$weight_avg"),
            ("winner_rate", "$winner_rate"),
        ]

        # Add raw factors (industry neutralization will be done via processor)
        # Industry neutralization: (factor - industry_mean) / industry_std, using pandas groupby(industry)
        # Note: Industry neutralization should be performed using IndustryNeutralizeProcessor
        # after data loading, as qlib doesn't support GroupNeutralize in expressions
        extra_fields = []
        extra_names = []
        
        for name, field in additional_factors:
            # Add raw factor field (industry neutralization will be applied later via processor)
            extra_fields.append(field)
            extra_names.append(name.upper())

        fields += extra_fields
        names += extra_names

        return fields, names
