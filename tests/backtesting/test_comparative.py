"""Tests for comparative runner and ablation experiment configs."""

import unittest

import pandas as pd

from tradingagents.backtesting.comparative import ComparativeResult, ExperimentConfig
from tradingagents.backtesting.config import BacktestConfig
from tradingagents.backtesting.results import BacktestResult
from tradingagents.backtesting.portfolio import DailySnapshot


def _make_result(total_return=0.05, sharpe=1.0, max_dd=-0.10):
    """Helper to create a BacktestResult with given metrics."""
    return BacktestResult(
        ticker="AAPL",
        start_date="2024-01-02",
        end_date="2024-06-01",
        initial_cash=100_000.0,
        metrics={
            "total_return": total_return,
            "annualized_return": total_return * 2,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sharpe * 1.2,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": 10,
            "calmar_ratio": abs(total_return / max_dd) if max_dd != 0 else 0,
            "win_rate": 0.55,
            "profit_factor": 1.3,
            "avg_win": 0.012,
            "avg_loss": -0.009,
            "total_trading_days": 120,
            "volatility_annualized": 0.15,
        },
        daily_snapshots=[
            DailySnapshot("2024-06-01", 105_000.0, 5_000.0, 100_000.0)
        ],
        elapsed_seconds=60.0,
    )


class TestComparativeResult(unittest.TestCase):
    def test_comparison_df_single_trial(self):
        """Single trial per experiment should work without std issues."""
        result = ComparativeResult()
        result.experiment_results = {
            "Full": [_make_result(0.10, 1.5, -0.08)],
            "No Debate": [_make_result(0.05, 0.8, -0.12)],
        }

        df = result.get_comparison_df()
        self.assertEqual(len(df), 2)
        self.assertIn("Full", df.index)
        self.assertIn("No Debate", df.index)
        self.assertAlmostEqual(df.loc["Full", "total_return_mean"], 0.10)
        self.assertAlmostEqual(df.loc["No Debate", "sharpe_ratio_mean"], 0.8)
        # Single trial -> std should be 0
        self.assertAlmostEqual(df.loc["Full", "total_return_std"], 0.0)

    def test_comparison_df_multiple_trials(self):
        """Multiple trials should produce non-zero std."""
        result = ComparativeResult()
        result.experiment_results = {
            "Full": [
                _make_result(0.10, 1.5, -0.08),
                _make_result(0.12, 1.6, -0.07),
                _make_result(0.08, 1.4, -0.09),
            ],
        }

        df = result.get_comparison_df()
        self.assertEqual(len(df), 1)
        # Mean should be average of [0.10, 0.12, 0.08] = 0.10
        self.assertAlmostEqual(df.loc["Full", "total_return_mean"], 0.10, places=4)
        # Std should be non-zero
        self.assertGreater(df.loc["Full", "total_return_std"], 0)

    def test_comparison_df_empty(self):
        result = ComparativeResult()
        df = result.get_comparison_df()
        self.assertTrue(df.empty)

    def test_comparison_text(self):
        result = ComparativeResult()
        result.experiment_results = {
            "Full": [_make_result(0.10, 1.5, -0.08)],
            "No Debate": [_make_result(0.05, 0.8, -0.12)],
        }

        text = result.generate_comparison_text()
        self.assertIn("COMPARATIVE BACKTEST RESULTS", text)
        self.assertIn("Full", text)
        self.assertIn("No Debate", text)
        self.assertIn("total_return", text)
        self.assertIn("sharpe_ratio", text)

    def test_comparison_text_empty(self):
        result = ComparativeResult()
        text = result.generate_comparison_text()
        self.assertEqual(text, "No results to compare.")


class TestExperimentConfig(unittest.TestCase):
    def test_default_values(self):
        config = ExperimentConfig(
            name="Test",
            backtest_config=BacktestConfig(),
        )
        self.assertEqual(config.name, "Test")
        self.assertEqual(config.num_trials, 1)

    def test_custom_trials(self):
        config = ExperimentConfig(
            name="Multi-trial",
            backtest_config=BacktestConfig(),
            num_trials=5,
        )
        self.assertEqual(config.num_trials, 5)


class TestAblationExperiments(unittest.TestCase):
    def test_creates_standard_experiments(self):
        from tradingagents.backtesting.ablation import create_ablation_experiments

        experiments = create_ablation_experiments(
            ticker="AAPL",
            start_date="2024-01-02",
            end_date="2024-06-01",
        )

        # Should create 6 standard experiments
        self.assertEqual(len(experiments), 6)
        names = [e.name for e in experiments]
        self.assertIn("Full Framework", names)
        self.assertIn("No Debate", names)
        self.assertIn("Market Analyst Only", names)
        self.assertIn("Fundamentals Only", names)
        self.assertIn("News Only", names)
        self.assertIn("No Reflection", names)

    def test_no_debate_config(self):
        from tradingagents.backtesting.ablation import create_ablation_experiments

        experiments = create_ablation_experiments(
            ticker="AAPL",
            start_date="2024-01-02",
            end_date="2024-06-01",
        )

        no_debate = [e for e in experiments if e.name == "No Debate"][0]
        self.assertEqual(
            no_debate.backtest_config.graph_config["max_debate_rounds"], 0
        )
        self.assertEqual(
            no_debate.backtest_config.graph_config["max_risk_discuss_rounds"], 0
        )

    def test_single_analyst_configs(self):
        from tradingagents.backtesting.ablation import create_ablation_experiments

        experiments = create_ablation_experiments(
            ticker="NVDA",
            start_date="2024-01-02",
            end_date="2024-06-01",
        )

        market_only = [e for e in experiments if e.name == "Market Analyst Only"][0]
        self.assertEqual(market_only.backtest_config.selected_analysts, ["market"])

        fund_only = [e for e in experiments if e.name == "Fundamentals Only"][0]
        self.assertEqual(fund_only.backtest_config.selected_analysts, ["fundamentals"])

    def test_custom_experiment(self):
        from tradingagents.backtesting.ablation import create_custom_experiment

        exp = create_custom_experiment(
            name="Custom Test",
            ticker="MSFT",
            start_date="2024-01-02",
            end_date="2024-03-01",
            selected_analysts=["market", "news"],
            num_trials=3,
        )
        self.assertEqual(exp.name, "Custom Test")
        self.assertEqual(exp.num_trials, 3)
        self.assertEqual(exp.backtest_config.ticker, "MSFT")
        self.assertEqual(exp.backtest_config.selected_analysts, ["market", "news"])

    def test_num_trials_passed_through(self):
        from tradingagents.backtesting.ablation import create_ablation_experiments

        experiments = create_ablation_experiments(
            ticker="AAPL",
            start_date="2024-01-02",
            end_date="2024-06-01",
            num_trials=3,
        )
        for exp in experiments:
            self.assertEqual(exp.num_trials, 3)


if __name__ == "__main__":
    unittest.main()
