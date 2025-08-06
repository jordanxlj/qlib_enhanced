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

    @patch("qlib.contrib.strategy.signal_strategy.portfolio_volatility")
    @patch("qlib.contrib.strategy.signal_strategy.VolTopkDropoutStrategy._compute_portfolio_volatility")
    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_volatility_adjustment(self, mock_create_signal, mock_compute_vol, mock_port_vol):
        mock_create_signal.return_value = self.signal
        mock_port_vol.return_value = pd.Series([0.035355], index=[pd.Timestamp("2023-01-03")])

        params = self.strategy_params.copy()
        params.update({"vol_window": 20, "target_volatility": 0.10, "position_weight": "inverse_vol"})
        strategy = self._create_strategy(params)

        mock_compute_vol.assert_called_once()

        stocks = [f"SH60000{i}" for i in range(5)]
        cov_data = {}
        for s1 in stocks:
            for s2 in stocks:
                cov_data[(pd.Timestamp("2023-01-03"), s1, s2)] = 0.0025 if s1 == s2 else 0.0
        cov_df = pd.DataFrame.from_dict(cov_data, orient="index", columns=["covariance"])
        cov_df.index = pd.MultiIndex.from_tuples(cov_df.index)
        strategy.vol_metrics = {
            "covariance": cov_df,
            "volatility": pd.DataFrame(0.05, index=[pd.Timestamp("2023-01-03")], columns=stocks),
        }

        strategy.order_generator = MagicMock()
        strategy.generate_trade_decision()

        args, kwargs = strategy.order_generator.generate_order_list_from_target_weight_position.call_args
        self.assertAlmostEqual(kwargs["risk_degree"], 0.178, places=3)

    @patch("qlib.contrib.strategy.signal_strategy.VolTopkDropoutStrategy._compute_portfolio_volatility")
    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_no_volatility_adjustment(self, mock_create_signal, mock_compute_vol):
        mock_create_signal.return_value = self.signal

        params = self.strategy_params.copy()
        params.update({"vol_window": 20, "position_weight": "average"})
        strategy = self._create_strategy(params)

        mock_compute_vol.assert_not_called()

        strategy.order_generator = MagicMock()
        strategy.generate_trade_decision()

        args, kwargs = strategy.order_generator.generate_order_list_from_target_weight_position.call_args
        self.assertEqual(kwargs["risk_degree"], 1.0)

    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_invalid_position_weight(self, mock_create_signal):
        mock_create_signal.return_value = self.signal

        params = self.strategy_params.copy()
        params.update({"position_weight": "invalid"})
        strategy = self._create_strategy(params)

        with self.assertRaises(ValueError) as cm:
            strategy.generate_trade_decision()
        self.assertEqual(str(cm.exception), "Invalid position_weight: invalid")

    @patch("qlib.contrib.strategy.signal_strategy.VolTopkDropoutStrategy._compute_portfolio_volatility")
    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_average_weight_with_vol_window(self, mock_create_signal, mock_compute_vol):
        mock_create_signal.return_value = self.signal

        params = self.strategy_params.copy()
        params.update({"vol_window": 20, "position_weight": "average"})
        strategy = self._create_strategy(params)

        mock_compute_vol.assert_not_called()

        strategy.order_generator = MagicMock()
        strategy.generate_trade_decision()

        args, kwargs = strategy.order_generator.generate_order_list_from_target_weight_position.call_args
        self.assertEqual(kwargs["risk_degree"], 1.0)

    @patch("qlib.contrib.strategy.signal_strategy.VolTopkDropoutStrategy._compute_portfolio_volatility")
    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_inverse_vol_weighting(self, mock_create_signal, mock_compute_vol):
        mock_create_signal.return_value = self.signal

        params = self.strategy_params.copy()
        params.update({"vol_window": 20, "position_weight": "inverse_vol"})
        strategy = self._create_strategy(params)

        mock_compute_vol.assert_called_once()

        stocks = [f"SH60000{i}" for i in range(3)]
        strategy.vol_metrics = {
            "volatility": pd.DataFrame(
                {"SH600000": 0.1, "SH600001": 0.2, "SH600002": 0.3},
                index=[pd.Timestamp("2023-01-03")]
            )
        }

        weights = strategy._calculate_weights(stocks, pd.Timestamp("2023-01-03"))

        total_inv = 1/0.1 + 1/0.2 + 1/0.3
        expected = {
            "SH600000": (1/0.1) / total_inv,
            "SH600001": (1/0.2) / total_inv,
            "SH600002": (1/0.3) / total_inv,
        }
        self.assertEqual(weights, expected)


if __name__ == "__main__":
    unittest.main()