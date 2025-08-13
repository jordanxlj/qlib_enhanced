# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from qlib.backtest.position import Position
from qlib.backtest.exchange import Exchange
from qlib.contrib.strategy.signal_strategy import VolTopkDropoutStrategy

class TestVolTopkDropoutStrategy(unittest.TestCase):
    def setUp(self):
        self.trade_calendar = MagicMock()
        self.trade_calendar.get_trade_step.return_value = 0
        self.trade_calendar.get_trade_len.return_value = 10

        def get_step_time_mock(*args, **kwargs):
            if kwargs.get("shift") == 1:
                return (pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03"))
            else:
                return (pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"))

        self.trade_calendar.get_step_time.side_effect = get_step_time_mock

        self.trade_exchange = MagicMock(spec=Exchange)
        self.trade_exchange.is_stock_tradable.return_value = True

        self.signal = MagicMock()
        self.signal.get_signal.return_value = pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5], index=[f"SH60000{i}" for i in range(5)], name="score"
        )

        self.trade_position = Position()
        trade_account = MagicMock()
        trade_account.current_position = self.trade_position
        self.common_infra = {"trade_account": trade_account}

        self.strategy_params = {
            "topk": 2,
            "n_drop": 1,
            "risk_degree": 1.0,
            "trade_exchange": self.trade_exchange,
        }

    def _create_strategy(self, params):
        strategy = VolTopkDropoutStrategy(signal=self.signal, **params)
        strategy.reset(
            level_infra={"trade_calendar": self.trade_calendar},
            common_infra=self.common_infra,
        )
        return strategy

    @patch(
        "qlib.contrib.strategy.signal_strategy.VolTopkDropoutStrategy._compute_portfolio_volatility"
    )
    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_volatility_quantile_weighting(
        self, mock_create_signal, mock_compute_vol
    ):
        mock_create_signal.return_value = self.signal

        params = self.strategy_params.copy()
        params.update({"vol_window": 20})
        strategy = self._create_strategy(params)

        mock_compute_vol.assert_called_once()

        # Create a more detailed volatility history to test trend filtering
        dates = pd.date_range(end="2023-01-03", periods=25)
        stocks = [f"SH60000{i}" for i in range(5)]
        vol_data = pd.DataFrame(index=dates, columns=stocks, dtype=float)

        # SH600000: Low volatility (included)
        vol_data["SH600000"] = -0.05
        # SH600001: High volatility, but upward trend (included)
        vol_data["SH600001"] = np.linspace(0.12, 0.15, 25)
        # SH600002: High volatility, downward trend (excluded)
        vol_data["SH600002"] = np.linspace(0.28, 0.25, 25)
        # SH600003: High volatility, no trend (excluded)
        vol_data["SH600003"] = 0.35
        # SH600004: Low volatility (included)
        vol_data["SH600004"] = 0.08

        strategy.vol_metrics = {"volatility": vol_data}

        # We are testing the logic for 2023-01-03
        weights = strategy._calculate_weights(stocks, pd.Timestamp("2023-01-03"))

        # Expected weights based on filtering and quantile allocation
        # Included: SH600000 (low_vol), SH600001 (up_trend), SH600004 (low_vol)
        # Excluded: SH600002 (down_trend), SH600003 (high_vol)
        expected = {
            # Group -10-10% (55% total): SH600000 (-0.05), SH600004 (0.08)
            "SH600000": 0.55 / 2,
            "SH600004": 0.55 / 2,
            # Group 10-20% (25% total): SH600001 (0.15)
            "SH600001": 0.25 / 1,
        }

        self.assertEqual(len(weights), 3)
        for stock, weight in expected.items():
            self.assertIn(stock, weights)
            self.assertAlmostEqual(weights[stock], weight)


if __name__ == "__main__":
    unittest.main()