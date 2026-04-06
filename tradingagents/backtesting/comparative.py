"""Comparative backtesting: run multiple configurations and compare results."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .config import BacktestConfig
from .results import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment in a comparative backtest."""

    name: str
    backtest_config: BacktestConfig
    num_trials: int = 1  # multiple trials to estimate LLM variance


@dataclass
class ComparativeResult:
    """Results from running multiple experiment configurations."""

    experiment_results: Dict[str, List[BacktestResult]] = field(
        default_factory=dict
    )

    def get_comparison_df(self) -> pd.DataFrame:
        """Build a comparison DataFrame: metrics x experiments (mean +/- std).

        Returns a DataFrame with experiment names as columns and metric names
        as rows. Each cell contains the mean value; if num_trials > 1,
        a separate std column is included.
        """
        rows = []
        metric_keys = None

        for name, trials in self.experiment_results.items():
            if not trials:
                continue

            if metric_keys is None:
                metric_keys = list(trials[0].metrics.keys())

            trial_metrics = [t.metrics for t in trials]
            row = {"experiment": name}

            for key in metric_keys:
                values = [m.get(key, 0.0) for m in trial_metrics]
                row[f"{key}_mean"] = sum(values) / len(values)
                if len(values) > 1:
                    mean = row[f"{key}_mean"]
                    variance = sum((v - mean) ** 2 for v in values) / (
                        len(values) - 1
                    )
                    row[f"{key}_std"] = variance**0.5
                else:
                    row[f"{key}_std"] = 0.0

            rows.append(row)

        return pd.DataFrame(rows).set_index("experiment") if rows else pd.DataFrame()

    def generate_comparison_text(self) -> str:
        """Generate a human-readable comparison table."""
        df = self.get_comparison_df()
        if df.empty:
            return "No results to compare."

        lines = ["=" * 80, "COMPARATIVE BACKTEST RESULTS", "=" * 80, ""]

        # Select key metrics for the summary table
        key_metrics = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "total_trading_days",
        ]

        for metric in key_metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col not in df.columns:
                continue

            lines.append(f"--- {metric} ---")
            for exp_name in df.index:
                mean_val = df.loc[exp_name, mean_col]
                std_val = df.loc[exp_name, std_col]

                if "return" in metric or "drawdown" in metric or "win_rate" in metric:
                    val_str = f"{mean_val:.2%}"
                    if std_val > 0:
                        val_str += f" (+/- {std_val:.2%})"
                else:
                    val_str = f"{mean_val:.4f}"
                    if std_val > 0:
                        val_str += f" (+/- {std_val:.4f})"

                lines.append(f"  {exp_name:30s} {val_str}")
            lines.append("")

        return "\n".join(lines)


class ComparativeRunner:
    """Runs multiple experiment configurations and compares results."""

    def __init__(self, experiments: List[ExperimentConfig]):
        self.experiments = experiments

    def run_all(self, progress_callback=None) -> ComparativeResult:
        """Run all experiments sequentially.

        Args:
            progress_callback: Optional callable(experiment_name, trial_num, total_trials)

        Returns:
            ComparativeResult with all experiment results.
        """
        from .runner import BacktestRunner

        result = ComparativeResult()

        for exp in self.experiments:
            logger.info(
                f"Running experiment: {exp.name} ({exp.num_trials} trial(s))"
            )
            trial_results = []

            for trial in range(exp.num_trials):
                logger.info(f"  Trial {trial + 1}/{exp.num_trials}")

                if progress_callback:
                    progress_callback(exp.name, trial + 1, exp.num_trials)

                # Disable caching between trials to capture LLM variance
                trial_config = BacktestConfig(
                    ticker=exp.backtest_config.ticker,
                    start_date=exp.backtest_config.start_date,
                    end_date=exp.backtest_config.end_date,
                    initial_cash=exp.backtest_config.initial_cash,
                    signal_allocation=exp.backtest_config.signal_allocation,
                    transaction_cost_bps=exp.backtest_config.transaction_cost_bps,
                    slippage_bps=exp.backtest_config.slippage_bps,
                    enable_reflection=exp.backtest_config.enable_reflection,
                    reflection_frequency=exp.backtest_config.reflection_frequency,
                    cache_analyst_outputs=False if exp.num_trials > 1 else exp.backtest_config.cache_analyst_outputs,
                    cache_dir=exp.backtest_config.cache_dir,
                    graph_config=exp.backtest_config.graph_config.copy(),
                    selected_analysts=list(exp.backtest_config.selected_analysts),
                    results_dir=exp.backtest_config.results_dir,
                    max_retries=exp.backtest_config.max_retries,
                )

                runner = BacktestRunner(trial_config)
                try:
                    trial_result = runner.run()
                    trial_results.append(trial_result)
                except Exception as e:
                    logger.error(
                        f"Trial {trial + 1} of {exp.name} failed: {e}"
                    )

            result.experiment_results[exp.name] = trial_results

        return result
