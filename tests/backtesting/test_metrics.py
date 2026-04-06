"""Tests for the metrics computation module."""

import unittest

import numpy as np
import pandas as pd

from tradingagents.backtesting.metrics import compute_metrics, _empty_metrics


class TestMetrics(unittest.TestCase):
    def test_empty_returns(self):
        result = compute_metrics(pd.Series(dtype=float))
        self.assertEqual(result["total_return"], 0.0)
        self.assertEqual(result["total_trading_days"], 0)

    def test_single_return(self):
        result = compute_metrics(pd.Series([0.01]))
        # Single return is not enough for meaningful stats, returns empty
        self.assertEqual(result["total_trading_days"], 0)

    def test_constant_positive_returns(self):
        """Daily 1% returns over 10 days."""
        returns = pd.Series([0.01] * 10)
        result = compute_metrics(returns, risk_free_rate=0.0)

        # Total return: (1.01)^10 - 1 ~ 10.46%
        self.assertAlmostEqual(result["total_return"], (1.01**10) - 1, places=4)
        self.assertEqual(result["win_rate"], 1.0)
        self.assertEqual(result["max_drawdown"], 0.0)
        self.assertGreater(result["sharpe_ratio"], 0)

    def test_constant_negative_returns(self):
        """Daily -1% returns over 10 days."""
        returns = pd.Series([-0.01] * 10)
        result = compute_metrics(returns, risk_free_rate=0.0)

        self.assertLess(result["total_return"], 0)
        self.assertEqual(result["win_rate"], 0.0)
        self.assertLess(result["max_drawdown"], 0)

    def test_mixed_returns(self):
        """Alternating positive and negative returns."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        result = compute_metrics(returns, risk_free_rate=0.0)

        self.assertGreater(result["total_return"], 0)
        self.assertGreater(result["win_rate"], 0)
        self.assertLess(result["win_rate"], 1.0)
        self.assertLess(result["max_drawdown"], 0)
        self.assertGreater(result["profit_factor"], 0)

    def test_max_drawdown_calculation(self):
        """Verify max drawdown with a known sequence."""
        # Up 10%, then down 20%, then up 5%
        returns = pd.Series([0.10, -0.20, 0.05])
        result = compute_metrics(returns, risk_free_rate=0.0)

        # After +10%: cum = 1.10
        # After -20%: cum = 1.10 * 0.80 = 0.88, drawdown from 1.10 = (0.88-1.10)/1.10 = -0.2
        self.assertAlmostEqual(result["max_drawdown"], -0.2, places=4)

    def test_sharpe_ratio_sign(self):
        """Positive returns with zero risk-free rate should give positive Sharpe."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01])
        result = compute_metrics(returns, risk_free_rate=0.0)
        self.assertGreater(result["sharpe_ratio"], 0)

    def test_sortino_ratio(self):
        """Sortino should be higher than Sharpe when gains > losses."""
        returns = pd.Series([0.05, -0.01, 0.04, -0.005, 0.03])
        result = compute_metrics(returns, risk_free_rate=0.0)
        # Sortino only penalizes downside, so should be >= Sharpe
        self.assertGreaterEqual(result["sortino_ratio"], result["sharpe_ratio"])

    def test_max_drawdown_duration(self):
        """Test drawdown duration tracking."""
        # Peak at day 1, drawdown for days 2-4, recovery at day 5
        returns = pd.Series([0.10, -0.05, -0.03, -0.02, 0.15])
        result = compute_metrics(returns, risk_free_rate=0.0)
        self.assertGreaterEqual(result["max_drawdown_duration_days"], 3)

    def test_empty_metrics_helper(self):
        result = _empty_metrics()
        self.assertEqual(result["total_return"], 0.0)
        self.assertEqual(result["total_trading_days"], 0)


if __name__ == "__main__":
    unittest.main()
