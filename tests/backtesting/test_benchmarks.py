"""Tests for benchmark strategies."""

import unittest

import numpy as np
import pandas as pd

from tradingagents.backtesting.benchmarks import (
    buy_and_hold,
    momentum_strategy,
    mean_reversion_strategy,
    _compute_rsi,
)
from tradingagents.backtesting.metrics import compute_metrics


class TestBuyAndHold(unittest.TestCase):
    def _make_prices(self, values):
        dates = pd.date_range("2024-01-02", periods=len(values), freq="B")
        return pd.Series(values, index=[d.strftime("%Y-%m-%d") for d in dates])

    def test_basic_profit(self):
        """Buy at $100, price rises to $110. Should profit."""
        prices = self._make_prices([100.0, 102.0, 105.0, 108.0, 110.0])
        snapshots = buy_and_hold(prices, initial_cash=10_000.0)

        self.assertEqual(len(snapshots), 5)
        # Started with 10k, bought 100 shares at $100
        self.assertAlmostEqual(snapshots[0].equity, 10_000.0, places=0)
        # Last day: 100 shares * $110 = $11,000
        self.assertGreater(snapshots[-1].equity, 10_000.0)

    def test_basic_loss(self):
        """Price drops. Should lose money."""
        prices = self._make_prices([100.0, 95.0, 90.0])
        snapshots = buy_and_hold(prices, initial_cash=10_000.0)
        self.assertLess(snapshots[-1].equity, 10_000.0)

    def test_empty_prices(self):
        snapshots = buy_and_hold(pd.Series(dtype=float))
        self.assertEqual(len(snapshots), 0)

    def test_equity_curve_monotonic_with_rising_prices(self):
        prices = self._make_prices([100.0 + i for i in range(20)])
        snapshots = buy_and_hold(prices, initial_cash=100_000.0)

        equities = [s.equity for s in snapshots]
        for i in range(1, len(equities)):
            self.assertGreaterEqual(equities[i], equities[i - 1])


class TestMomentumStrategy(unittest.TestCase):
    def _make_prices(self, values):
        dates = pd.date_range("2024-01-02", periods=len(values), freq="B")
        return pd.Series(values, index=[d.strftime("%Y-%m-%d") for d in dates])

    def test_uptrend_buys(self):
        """Steady uptrend should trigger buy after lookback period."""
        prices = self._make_prices([100 + i * 0.5 for i in range(30)])
        snapshots = momentum_strategy(prices, lookback=10, initial_cash=100_000.0)

        self.assertEqual(len(snapshots), 30)
        # Should have bought at some point
        has_position = any(s.position_value > 0 for s in snapshots)
        self.assertTrue(has_position)

    def test_downtrend_stays_out(self):
        """Steady downtrend should not hold position after selling."""
        prices = self._make_prices([100 - i * 0.5 for i in range(30)])
        snapshots = momentum_strategy(prices, lookback=10, initial_cash=100_000.0)

        # After the lookback, should be out of the market
        late_snapshots = snapshots[15:]
        all_out = all(s.position_value == 0 for s in late_snapshots)
        self.assertTrue(all_out)

    def test_short_series_falls_back(self):
        """Series shorter than lookback should fall back to buy-and-hold."""
        prices = self._make_prices([100, 101, 102])
        snapshots = momentum_strategy(prices, lookback=10)
        self.assertEqual(len(snapshots), 3)


class TestMeanReversionStrategy(unittest.TestCase):
    def _make_prices(self, values):
        dates = pd.date_range("2024-01-02", periods=len(values), freq="B")
        return pd.Series(values, index=[d.strftime("%Y-%m-%d") for d in dates])

    def test_produces_snapshots(self):
        np.random.seed(42)
        base = 100.0
        returns = np.random.normal(0, 0.02, 50)
        prices = base * np.cumprod(1 + returns)
        price_series = self._make_prices(prices.tolist())

        snapshots = mean_reversion_strategy(price_series, rsi_period=14)
        self.assertEqual(len(snapshots), 50)

    def test_short_series(self):
        prices = self._make_prices([100, 101, 102])
        snapshots = mean_reversion_strategy(prices, rsi_period=14)
        self.assertEqual(len(snapshots), 3)


class TestRSI(unittest.TestCase):
    def test_rsi_range(self):
        """RSI should always be between 0 and 100."""
        np.random.seed(123)
        prices = 100.0 * np.cumprod(1 + np.random.normal(0, 0.02, 50))
        rsi = _compute_rsi(prices, 14)

        for val in rsi:
            if val is not None:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 100.0)

    def test_rsi_all_gains(self):
        """All positive returns should give RSI close to 100."""
        prices = np.array([100 + i for i in range(20)], dtype=float)
        rsi = _compute_rsi(prices, 14)
        # Last RSI should be 100 (all gains, no losses)
        self.assertAlmostEqual(rsi[-1], 100.0, places=1)

    def test_rsi_all_losses(self):
        """All negative returns should give RSI close to 0."""
        prices = np.array([100 - i * 0.5 for i in range(20)], dtype=float)
        rsi = _compute_rsi(prices, 14)
        self.assertAlmostEqual(rsi[-1], 0.0, places=1)


class TestBenchmarkMetricsCompatibility(unittest.TestCase):
    """Verify benchmark snapshots work with compute_metrics()."""

    def test_buy_and_hold_metrics(self):
        dates = pd.date_range("2024-01-02", periods=20, freq="B")
        prices = pd.Series(
            [100 + i * 0.5 for i in range(20)],
            index=[d.strftime("%Y-%m-%d") for d in dates],
        )
        snapshots = buy_and_hold(prices, initial_cash=100_000.0)

        equities = pd.Series(
            [s.equity for s in snapshots],
            index=[s.date for s in snapshots],
        )
        returns = equities.pct_change().dropna()
        metrics = compute_metrics(returns)

        self.assertIn("total_return", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertGreater(metrics["total_return"], 0)


if __name__ == "__main__":
    unittest.main()
