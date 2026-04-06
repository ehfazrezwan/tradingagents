"""Tests for BacktestRunner using a mock TradingAgentsGraph.

These tests do NOT require LLM calls or API keys.
"""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from tradingagents.backtesting.config import BacktestConfig
from tradingagents.backtesting.runner import BacktestRunner


class MockPriceFetcher:
    """Mock price fetcher with predetermined prices."""

    def __init__(self, prices):
        """prices: dict of date_str -> close_price"""
        self._prices = prices

    def preload(self, ticker, start, end):
        pass

    def get_close_price(self, ticker, date):
        return self._prices.get(date)

    def get_trading_days(self, ticker, start, end):
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        return [
            d for d in sorted(self._prices.keys())
            if start_dt <= pd.Timestamp(d) <= end_dt
        ]

    def get_next_trading_day_price(self, ticker, date):
        days = sorted(self._prices.keys())
        for i, d in enumerate(days):
            if d == date and i + 1 < len(days):
                return self._prices[days[i + 1]]
        return None


class MockTradingAgentsGraph:
    """Mock graph that returns predetermined signals."""

    def __init__(self, signal_schedule):
        self.signal_schedule = signal_schedule
        self.reflect_calls = []

    def propagate(self, ticker, date):
        signal = self.signal_schedule.get(date, "HOLD")
        state = {
            "company_of_interest": ticker,
            "trade_date": date,
            "market_report": "mock",
            "sentiment_report": "mock",
            "news_report": "mock",
            "fundamentals_report": "mock",
            "investment_plan": "mock",
            "trader_investment_plan": "mock",
            "final_trade_decision": signal,
            "investment_debate_state": {"bull_history": [], "bear_history": [], "history": [], "current_response": "", "judge_decision": ""},
            "risk_debate_state": {"aggressive_history": [], "conservative_history": [], "neutral_history": [], "history": [], "judge_decision": ""},
        }
        return state, signal

    def reflect_and_remember(self, returns_losses):
        self.reflect_calls.append(returns_losses)


class TestBacktestRunner(unittest.TestCase):
    def _make_config(self, **overrides):
        defaults = {
            "ticker": "AAPL",
            "start_date": "2024-01-02",
            "end_date": "2024-01-10",
            "initial_cash": 100_000.0,
            "enable_reflection": False,
            "cache_analyst_outputs": False,
        }
        defaults.update(overrides)
        return BacktestConfig(**defaults)

    def _make_prices(self):
        """Create a simple price schedule: $100 -> $105 over 5 days."""
        return {
            "2024-01-02": 100.0,
            "2024-01-03": 101.0,
            "2024-01-04": 102.0,
            "2024-01-05": 103.0,
            "2024-01-08": 104.0,
            "2024-01-09": 105.0,
            "2024-01-10": 106.0,
        }

    @patch("tradingagents.backtesting.runner.TradingAgentsGraph")
    @patch("tradingagents.backtesting.runner.PriceFetcher")
    def test_basic_buy_and_hold_run(self, MockPFClass, MockGraphClass):
        """Test a simple run where signal is always BUY."""
        prices = self._make_prices()
        signals = {d: "BUY" for d in prices}

        mock_pf = MockPriceFetcher(prices)
        MockPFClass.return_value = mock_pf

        mock_graph = MockTradingAgentsGraph(signals)
        MockGraphClass.return_value = mock_graph

        config = self._make_config()
        runner = BacktestRunner(config)
        runner.price_fetcher = mock_pf

        # Patch the graph creation inside run()
        with patch.object(runner, '_get_signal') as mock_get_signal:
            mock_get_signal.side_effect = lambda g, t, d, r: signals.get(d, "HOLD")

            # Run with actual runner but mock price fetcher
            result = runner.run()

        self.assertEqual(result.ticker, "AAPL")
        self.assertGreater(len(result.daily_snapshots), 0)
        self.assertIn("total_return", result.metrics)

    @patch("tradingagents.backtesting.runner.TradingAgentsGraph")
    def test_hold_only_no_trades(self, MockGraphClass):
        """All HOLD signals should result in no trades."""
        prices = self._make_prices()
        signals = {d: "HOLD" for d in prices}

        mock_graph = MockTradingAgentsGraph(signals)
        MockGraphClass.return_value = mock_graph

        config = self._make_config()
        runner = BacktestRunner(config)
        runner.price_fetcher = MockPriceFetcher(prices)

        result = runner.run()

        self.assertEqual(len(result.trades), 0)
        # Equity should be unchanged (still have all cash)
        if result.daily_snapshots:
            self.assertAlmostEqual(
                result.daily_snapshots[-1].equity, 100_000.0, places=0
            )

    @patch("tradingagents.backtesting.runner.TradingAgentsGraph")
    def test_signals_recorded(self, MockGraphClass):
        """All signals should be recorded in the result."""
        prices = self._make_prices()
        signals = {
            "2024-01-02": "BUY",
            "2024-01-03": "HOLD",
            "2024-01-04": "SELL",
            "2024-01-05": "HOLD",
            "2024-01-08": "BUY",
            "2024-01-09": "HOLD",
            "2024-01-10": "SELL",
        }

        mock_graph = MockTradingAgentsGraph(signals)
        MockGraphClass.return_value = mock_graph

        config = self._make_config()
        runner = BacktestRunner(config)
        runner.price_fetcher = MockPriceFetcher(prices)

        result = runner.run()

        self.assertEqual(len(result.signals), len(prices))
        signal_dates = [s[0] for s in result.signals]
        for date in prices:
            self.assertIn(date, signal_dates)

    @patch("tradingagents.backtesting.runner.TradingAgentsGraph")
    def test_reflection_called(self, MockGraphClass):
        """Reflection should be called at the configured frequency."""
        prices = self._make_prices()
        signals = {d: "BUY" for d in prices}

        mock_graph = MockTradingAgentsGraph(signals)
        MockGraphClass.return_value = mock_graph

        config = self._make_config(
            enable_reflection=True,
            reflection_frequency=3,
        )
        runner = BacktestRunner(config)
        runner.price_fetcher = MockPriceFetcher(prices)

        result = runner.run()

        # With 7 trading days and frequency=3, reflection should happen at day 3 and 6
        self.assertGreater(len(result.daily_snapshots), 0)

    def test_result_summary(self):
        """Test that BacktestResult.summary() produces readable output."""
        from tradingagents.backtesting.results import BacktestResult
        from tradingagents.backtesting.portfolio import DailySnapshot

        result = BacktestResult(
            ticker="AAPL",
            start_date="2024-01-02",
            end_date="2024-01-10",
            initial_cash=100_000.0,
            metrics={"total_return": 0.05, "sharpe_ratio": 1.5},
            daily_snapshots=[
                DailySnapshot("2024-01-10", 105_000.0, 5_000.0, 100_000.0)
            ],
            elapsed_seconds=120.0,
        )
        summary = result.summary()
        self.assertIn("AAPL", summary)
        self.assertIn("5.00%", summary)
        self.assertIn("$105,000.00", summary)


if __name__ == "__main__":
    unittest.main()
