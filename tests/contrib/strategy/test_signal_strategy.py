# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from qlib.backtest.position import Position
from qlib.backtest.exchange import Exchange
from qlib.contrib.strategy.signal_strategy import VolTopkDropoutStrategy, OptimizedTopkDropoutStrategy
from qlib.backtest.decision import Order

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


class TestOptimizedTopkDropoutStrategy(unittest.TestCase):
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
        # default deal price and factor
        self.trade_exchange.get_deal_price.return_value = 10.0
        self.trade_exchange.get_factor.return_value = 1.0
        self.trade_exchange.check_order.return_value = True

        def deal_order_side_effect(order, position):
            price = 10.0
            trade_val = order.amount * price
            cost = 0.0
            return trade_val, cost, price

        self.trade_exchange.deal_order.side_effect = deal_order_side_effect

        self.signal = MagicMock()
        # simple increasing scores; will prefer buying new ones
        self.signal.get_signal.return_value = pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5], index=["AAA", "BBB", "CCC", "DDD", "EEE"], name="score"
        )

        # Start with one holding 'AAA' 50 shares @ 10, cash 500
        self.start_pos = Position(cash=500.0, position_dict={"AAA": {"amount": 50, "price": 10.0}})
        trade_account = MagicMock()
        trade_account.current_position = self.start_pos
        self.common_infra = {"trade_account": trade_account}

        self.strategy_params = {
            "topk": 2,
            "n_drop": 1,
            "risk_degree": 1.0,
            "trade_exchange": self.trade_exchange,
            "pm_lookback": 10,
        }

    def _make_strategy(self, position_manager=None):
        strategy = OptimizedTopkDropoutStrategy(signal=self.signal, **self.strategy_params)
        if position_manager is not None:
            strategy.set_position_manager(position_manager)
        strategy.reset(
            level_infra={"trade_calendar": self.trade_calendar},
            common_infra=self.common_infra,
        )
        return strategy

    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_rebalance_with_position_manager(self, mock_create_signal):
        mock_create_signal.return_value = self.signal

        # Price cache for returns
        dates = pd.date_range("2022-12-20", periods=12, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": 10 + np.arange(len(dates)) * 0,  # flat
                "BBB": 10 + np.arange(len(dates)) * 0.1,
                "CCC": 10 + np.arange(len(dates)) * 0.05,
            },
            index=dates,
        )

        pm = MagicMock()
        # Optimizer wants 30% AAA, 70% BBB across union holdings+buy candidates
        pm.optimize.return_value = pd.Series({"AAA": 0.3, "BBB": 0.7, "CCC": 0.0})

        strategy = self._make_strategy(position_manager=pm)
        strategy._pm_price_cache = prices

        decision = strategy.generate_trade_decision()
        orders = decision.get_decision()
        # Expect one partial sell of AAA (reduce from ~$500 to $300), and one buy of BBB (~$700)
        sell_orders = [o for o in orders if o.direction == Order.SELL]
        buy_orders = [o for o in orders if o.direction == Order.BUY]

        self.assertGreaterEqual(len(sell_orders), 1)
        self.assertGreaterEqual(len(buy_orders), 1)

        # Check approximate amounts given price 10
        total_value = 500.0 + 500.0  # stock value + cash
        target_aaa_val = 0.3 * total_value
        need_sell_val = max(0.0, 500.0 - target_aaa_val)
        # aggregate sell amounts for AAA
        sold_units = sum(o.amount for o in sell_orders if o.stock_id == "AAA")
        self.assertAlmostEqual(sold_units * 10.0, need_sell_val, delta=1e-6)

    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_fallback_without_position_manager_equal_weight(self, mock_create_signal):
        mock_create_signal.return_value = self.signal

        strategy = self._make_strategy(position_manager=None)
        # No price cache needed since PM not used; should equal weight across union
        decision = strategy.generate_trade_decision()
        orders = decision.get_decision()
        self.assertTrue(any(o.direction == Order.BUY for o in orders))

    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_returns_with_missing_data_cleaning(self, mock_create_signal):
        mock_create_signal.return_value = self.signal

        # Create returns with many NaNs so only certain columns have sufficient observations
        dates = pd.date_range("2022-12-20", periods=12, freq="B")
        prices = pd.DataFrame(index=dates, columns=["AAA", "BBB", "CCC"], dtype=float)
        prices["AAA"] = 10.0  # constant available
        prices["BBB"] = np.where(np.arange(len(dates)) % 2 == 0, 10.0 + np.arange(len(dates)) * 0.1, np.nan)
        prices["CCC"] = np.nan  # all NaN -> should be dropped

        captured_returns = {}

        def pm_optimize_capture(returns_df):
            captured_returns["cols"] = list(returns_df.columns)
            # return equal weights for provided
            return pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns)

        pm = MagicMock()
        pm.optimize.side_effect = pm_optimize_capture

        strategy = self._make_strategy(position_manager=pm)
        strategy._pm_price_cache = prices

        strategy.generate_trade_decision()

        # Expect 'AAA' and possibly 'BBB' (depending on min_obs) but not 'CCC'
        self.assertIn("AAA", captured_returns.get("cols", []))
        self.assertNotIn("CCC", captured_returns.get("cols", []))

    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_not_tradable_skips_orders_and_cash_scaling(self, mock_create_signal):
        mock_create_signal.return_value = self.signal

        # Make only buys tradable for BBB; and set cash small to enforce scaling
        def is_tradable(stock_id, start_time, end_time, direction=None):
            if direction == Order.SELL and stock_id == "AAA":
                return True
            if direction == Order.BUY and stock_id == "BBB":
                return True
            return False

        self.trade_exchange.is_stock_tradable.side_effect = is_tradable
        # Reduce starting cash to force scaling
        self.start_pos.position["cash"] = 50.0

        dates = pd.date_range("2022-12-20", periods=12, freq="B")
        prices = pd.DataFrame({"AAA": 10.0, "BBB": 10.0}, index=dates)

        pm = MagicMock()
        pm.optimize.return_value = pd.Series({"AAA": 0.2, "BBB": 0.8})

        strategy = self._make_strategy(position_manager=pm)
        strategy._pm_price_cache = prices

        decision = strategy.generate_trade_decision()
        orders = decision.get_decision()

        # Only one buy for BBB and possibly one sell for AAA
        buy_orders = [o for o in orders if o.direction == Order.BUY]
        sell_orders = [o for o in orders if o.direction == Order.SELL]
        self.assertTrue(all(o.stock_id == "BBB" for o in buy_orders))
        # Verify buy not exceeding available cash (50)
        total_buy_val = sum(o.amount * 10.0 for o in buy_orders)
        self.assertLessEqual(total_buy_val, 50.0 + 1e-6)

    @patch("qlib.contrib.strategy.signal_strategy.create_signal_from")
    def test_partial_sell_uses_sell_price_and_no_oversell(self, mock_create_signal):
        mock_create_signal.return_value = self.signal

        # Start with only AAA holding: 500 shares at stored price 10.0, no cash
        self.start_pos.position["AAA"]["amount"] = 500.0
        self.start_pos.position["AAA"]["price"] = 10.0
        self.start_pos.position["cash"] = 0.0

        # Exchange sell price differs from stored price to simulate mismatch
        def get_deal_price_side(stock_id, start_time, end_time, direction):
            if direction == Order.SELL:
                return 10.05  # slightly different from stored 10.0
            return 10.0

        self.trade_exchange.get_deal_price.side_effect = get_deal_price_side
        self.trade_exchange.get_factor.return_value = 1.0
        self.trade_exchange.check_order.return_value = True

        def deal_order_side_effect(order, position):
            price = get_deal_price_side(order.stock_id, None, None, order.direction)
            trade_val = order.amount * price
            return trade_val, 0.0, price

        self.trade_exchange.deal_order.side_effect = deal_order_side_effect

        # PM targets 30% AAA, 70% BBB to force a sell in AAA
        pm = MagicMock()
        pm.optimize.return_value = pd.Series({"AAA": 0.3, "BBB": 0.7})

        # Provide price cache for AAA/BBB so PM universe includes both
        dates = pd.date_range("2022-12-20", periods=12, freq="B")
        prices = pd.DataFrame({"AAA": 10.0, "BBB": 10.0}, index=dates)

        strategy = self._make_strategy(position_manager=pm)
        strategy._pm_price_cache = prices

        # Prevent buys (BBB) from executing to isolate sell sizing behavior
        def is_tradable(stock_id, start_time, end_time, direction=None):
            if direction == Order.BUY:
                return False
            return True

        self.trade_exchange.is_stock_tradable.side_effect = is_tradable

        decision = strategy.generate_trade_decision()
        orders = decision.get_decision()

        sell_orders = [o for o in orders if o.direction == Order.SELL and o.stock_id == "AAA"]
        self.assertGreaterEqual(len(sell_orders), 1)

        # Compute expected sell shares using sell price to avoid oversell
        total_value = 500.0 * 10.0  # no cash
        target_val = 0.3 * total_value
        sell_price = 10.05
        expected_sell_shares = 500.0 - (target_val / sell_price)

        sold_units = sum(o.amount for o in sell_orders)
        # No oversell
        self.assertLessEqual(sold_units, 500.0 + 1e-9)
        # Close to expected computed with sell price
        self.assertAlmostEqual(sold_units, expected_sell_shares, places=6)


if __name__ == "__main__":
    unittest.main()