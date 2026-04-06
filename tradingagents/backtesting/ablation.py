"""Factory functions for standard ablation experiments.

Creates pre-configured ExperimentConfig sets to test whether
multi-agent debate adds value over simpler approaches.
"""

from typing import Dict, List

from tradingagents.default_config import DEFAULT_CONFIG

from .comparative import ExperimentConfig
from .config import BacktestConfig


def create_ablation_experiments(
    ticker: str,
    start_date: str,
    end_date: str,
    base_graph_config: Dict = None,
    num_trials: int = 1,
    initial_cash: float = 100_000.0,
) -> List[ExperimentConfig]:
    """Create standard ablation experiments for comparative testing.

    Generates experiments that test the value of:
    1. Full framework (all analysts + debate) — baseline
    2. No debate (max_debate_rounds=0, max_risk_discuss_rounds=0)
    3. Single analyst variants (market only, fundamentals only, news only)
    4. No reflection (enable_reflection=False)

    Args:
        ticker: Stock ticker to backtest.
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD).
        base_graph_config: Base TradingAgentsGraph config. Defaults to DEFAULT_CONFIG.
        num_trials: Number of trials per experiment (for variance estimation).
        initial_cash: Starting portfolio value.

    Returns:
        List of ExperimentConfig objects ready for ComparativeRunner.
    """
    if base_graph_config is None:
        base_graph_config = DEFAULT_CONFIG.copy()

    experiments = []

    def _make_config(
        selected_analysts=None,
        graph_overrides=None,
        enable_reflection=True,
    ) -> BacktestConfig:
        gc = base_graph_config.copy()
        if graph_overrides:
            gc.update(graph_overrides)
        return BacktestConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            selected_analysts=selected_analysts
            or ["market", "social", "news", "fundamentals"],
            graph_config=gc,
            enable_reflection=enable_reflection,
        )

    # 1. Full framework (baseline)
    experiments.append(
        ExperimentConfig(
            name="Full Framework",
            backtest_config=_make_config(),
            num_trials=num_trials,
        )
    )

    # 2. No debate
    experiments.append(
        ExperimentConfig(
            name="No Debate",
            backtest_config=_make_config(
                graph_overrides={
                    "max_debate_rounds": 0,
                    "max_risk_discuss_rounds": 0,
                }
            ),
            num_trials=num_trials,
        )
    )

    # 3. Single analyst: Market only
    experiments.append(
        ExperimentConfig(
            name="Market Analyst Only",
            backtest_config=_make_config(selected_analysts=["market"]),
            num_trials=num_trials,
        )
    )

    # 4. Single analyst: Fundamentals only
    experiments.append(
        ExperimentConfig(
            name="Fundamentals Only",
            backtest_config=_make_config(selected_analysts=["fundamentals"]),
            num_trials=num_trials,
        )
    )

    # 5. Single analyst: News only
    experiments.append(
        ExperimentConfig(
            name="News Only",
            backtest_config=_make_config(selected_analysts=["news"]),
            num_trials=num_trials,
        )
    )

    # 6. No reflection (no learning between days)
    experiments.append(
        ExperimentConfig(
            name="No Reflection",
            backtest_config=_make_config(enable_reflection=False),
            num_trials=num_trials,
        )
    )

    return experiments


def create_custom_experiment(
    name: str,
    ticker: str,
    start_date: str,
    end_date: str,
    selected_analysts: List[str],
    graph_config: Dict = None,
    enable_reflection: bool = True,
    num_trials: int = 1,
    initial_cash: float = 100_000.0,
) -> ExperimentConfig:
    """Create a single custom experiment configuration.

    Useful for ad-hoc experiments beyond the standard ablation set.
    """
    if graph_config is None:
        graph_config = DEFAULT_CONFIG.copy()

    return ExperimentConfig(
        name=name,
        backtest_config=BacktestConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            selected_analysts=selected_analysts,
            graph_config=graph_config,
            enable_reflection=enable_reflection,
        ),
        num_trials=num_trials,
    )
