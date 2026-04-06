"""Tests for the Portfolio class and supporting dataclasses."""

import unittest

from tradingagents.backtesting.portfolio import (
    DailySnapshot,
    Portfolio,
    Position,
    Trade,
)


class TestPosition(unittest.TestCase):
    def test_value_at(self):
        pos = Position(shares=10, avg_cost=100.0)
        self.assertAlmostEqual(pos.value_at(150.0), 1500.0)

    def test_empty_position_value(self):
        pos = Position()
        self.assertAlmostEqual(pos.value_at(100.0), 0.0)


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(
            initial_cash=100_000.0,
            transaction_cost_bps=10.0,
            slippage_bps=5.0,
        )
        self.alloc_map = {
            "BUY": 1.0,
            "OVERWEIGHT": 0.75,
            "HOLD": None,
            "UNDERWEIGHT": 0.25,
            "SELL": 0.0,
        }

    def test_initial_state(self):
        self.assertEqual(self.portfolio.cash, 100_000.0)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.trade_log), 0)

    def test_buy_signal(self):
        trade = self.portfolio.execute_signal(
            ticker="AAPL",
            signal="BUY",
            current_price=100.0,
            date="2024-01-02",
            allocation_map=self.alloc_map,
        )
        self.assertIsNotNone(trade)
        self.assertEqual(trade.action, "BUY")
        self.assertGreater(trade.shares, 0)
        self.assertLess(self.portfolio.cash, 100_000.0)
        self.assertGreater(self.portfolio.positions["AAPL"].shares, 0)

    def test_hold_signal_no_trade(self):
        trade = self.portfolio.execute_signal(
            ticker="AAPL",
            signal="HOLD",
            current_price=100.0,
            date="2024-01-02",
            allocation_map=self.alloc_map,
        )
        self.assertIsNone(trade)
        self.assertEqual(self.portfolio.cash, 100_000.0)

    def test_buy_then_sell(self):
        # Buy
        self.portfolio.execute_signal(
            "AAPL", "BUY", 100.0, "2024-01-02", self.alloc_map
        )
        shares_after_buy = self.portfolio.positions["AAPL"].shares
        self.assertGreater(shares_after_buy, 0)

        # Sell all
        self.portfolio.execute_signal(
            "AAPL", "SELL", 110.0, "2024-01-03", self.alloc_map
        )
        self.assertEqual(self.portfolio.positions["AAPL"].shares, 0)
        # Should have made money (bought at 100, sold at 110)
        self.assertGreater(self.portfolio.cash, 100_000.0)

    def test_transaction_costs_deducted(self):
        trade = self.portfolio.execute_signal(
            "AAPL", "BUY", 100.0, "2024-01-02", self.alloc_map
        )
        self.assertIsNotNone(trade)
        self.assertGreater(trade.cost, 0)
        # Cost = (10 + 5) / 10000 * shares * price = 0.15% of notional
        expected_cost_rate = (10.0 + 5.0) / 10_000
        expected_cost = trade.shares * 100.0 * expected_cost_rate
        self.assertAlmostEqual(trade.cost, expected_cost, places=2)

    def test_overweight_partial_allocation(self):
        # First buy full
        self.portfolio.execute_signal(
            "AAPL", "BUY", 100.0, "2024-01-02", self.alloc_map
        )
        full_shares = self.portfolio.positions["AAPL"].shares

        # Now set to UNDERWEIGHT (25% of equity)
        trade = self.portfolio.execute_signal(
            "AAPL", "UNDERWEIGHT", 100.0, "2024-01-03", self.alloc_map
        )
        self.assertIsNotNone(trade)
        self.assertEqual(trade.action, "SELL")
        self.assertLess(self.portfolio.positions["AAPL"].shares, full_shares)

    def test_get_equity(self):
        self.portfolio.execute_signal(
            "AAPL", "BUY", 100.0, "2024-01-02", self.alloc_map
        )
        equity = self.portfolio.get_equity({"AAPL": 110.0})
        # Equity should be close to initial + gains - costs
        self.assertGreater(equity, 0)

    def test_snapshot(self):
        self.portfolio.execute_signal(
            "AAPL", "BUY", 100.0, "2024-01-02", self.alloc_map
        )
        snap = self.portfolio.snapshot("2024-01-02", {"AAPL": 100.0}, signal="BUY")

        self.assertEqual(snap.date, "2024-01-02")
        self.assertEqual(snap.signal, "BUY")
        self.assertGreater(snap.equity, 0)
        self.assertEqual(len(self.portfolio.daily_snapshots), 1)

    def test_returns_series(self):
        # Create two snapshots with different equities
        self.portfolio.daily_snapshots.append(
            DailySnapshot(date="2024-01-02", equity=100_000, cash=0, position_value=100_000)
        )
        self.portfolio.daily_snapshots.append(
            DailySnapshot(date="2024-01-03", equity=101_000, cash=0, position_value=101_000)
        )
        self.portfolio.daily_snapshots.append(
            DailySnapshot(date="2024-01-04", equity=99_000, cash=0, position_value=99_000)
        )

        returns = self.portfolio.get_returns_series()
        self.assertEqual(len(returns), 2)
        self.assertAlmostEqual(returns.iloc[0], 0.01, places=4)  # +1%

    def test_cannot_buy_more_than_cash(self):
        """Portfolio should limit buys to available cash."""
        self.portfolio.cash = 500.0  # very limited cash
        trade = self.portfolio.execute_signal(
            "AAPL", "BUY", 100.0, "2024-01-02", self.alloc_map
        )
        if trade:
            total_spent = trade.shares * trade.price + trade.cost
            self.assertLessEqual(total_spent, 500.0 + 1.0)  # allow tiny float error


if __name__ == "__main__":
    unittest.main()
