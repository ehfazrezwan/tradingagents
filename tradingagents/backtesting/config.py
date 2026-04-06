from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tradingagents.default_config import DEFAULT_CONFIG


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    # Core parameters
    ticker: str = "AAPL"
    start_date: str = "2024-01-02"
    end_date: str = "2025-01-02"
    initial_cash: float = 100_000.0

    # Position sizing: maps signal -> target portfolio allocation (0.0-1.0)
    # None means no change (HOLD)
    signal_allocation: Dict[str, Optional[float]] = field(
        default_factory=lambda: {
            "BUY": 1.0,
            "OVERWEIGHT": 0.75,
            "HOLD": None,
            "UNDERWEIGHT": 0.25,
            "SELL": 0.0,
        }
    )

    # Transaction costs
    transaction_cost_bps: float = 10.0  # basis points per trade
    slippage_bps: float = 5.0  # basis points slippage

    # Reflection settings
    enable_reflection: bool = True
    reflection_frequency: int = 5  # reflect every N trading days

    # Caching
    cache_analyst_outputs: bool = True
    cache_dir: str = "./backtest_cache"

    # TradingAgentsGraph configuration (passed through)
    graph_config: Dict = field(default_factory=lambda: DEFAULT_CONFIG.copy())
    selected_analysts: List[str] = field(
        default_factory=lambda: ["market", "social", "news", "fundamentals"]
    )

    # Output
    results_dir: str = "./backtest_results"

    # Error handling
    max_retries: int = 2  # retries per trading day on propagate failure
