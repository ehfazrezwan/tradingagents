import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .portfolio import DailySnapshot, Trade


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""

    # Config summary (serializable subset)
    ticker: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_cash: float = 100_000.0

    # Performance
    metrics: Dict = field(default_factory=dict)
    benchmark_metrics: Dict = field(default_factory=dict)

    # Detailed data
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    signals: List[Tuple[str, str]] = field(default_factory=list)  # (date, signal)
    failed_dates: List[str] = field(default_factory=list)

    # Timing
    elapsed_seconds: float = 0.0

    def to_json(self, path: str) -> None:
        """Save full result to a JSON file."""
        directory = Path(path).parent
        directory.mkdir(parents=True, exist_ok=True)

        data = {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_cash": self.initial_cash,
            "metrics": self.metrics,
            "benchmark_metrics": self.benchmark_metrics,
            "signals": self.signals,
            "trades": [asdict(t) for t in self.trades],
            "failed_dates": self.failed_dates,
            "elapsed_seconds": self.elapsed_seconds,
            "daily_snapshots": [asdict(s) for s in self.daily_snapshots],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def to_csv(self, path: str) -> None:
        """Save the equity curve as CSV."""
        directory = Path(path).parent
        directory.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "date": s.date,
                "equity": s.equity,
                "cash": s.cash,
                "position_value": s.position_value,
                "signal": s.signal,
            }
            for s in self.daily_snapshots
        ]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def summary(self) -> str:
        """Human-readable text summary of the backtest."""
        lines = [
            f"Backtest Results: {self.ticker}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Duration: {self.elapsed_seconds:.1f}s",
            "",
            "--- Strategy Metrics ---",
        ]

        for key, val in self.metrics.items():
            if isinstance(val, float):
                if "return" in key or "drawdown" in key or "win_rate" in key:
                    lines.append(f"  {key}: {val:.2%}")
                else:
                    lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")

        if self.benchmark_metrics:
            lines.append("")
            lines.append("--- Buy & Hold Benchmark ---")
            for key, val in self.benchmark_metrics.items():
                if isinstance(val, float):
                    if "return" in key or "drawdown" in key or "win_rate" in key:
                        lines.append(f"  {key}: {val:.2%}")
                    else:
                        lines.append(f"  {key}: {val:.4f}")
                else:
                    lines.append(f"  {key}: {val}")

        lines.append("")
        lines.append(f"Total trades: {len(self.trades)}")
        lines.append(f"Failed dates: {len(self.failed_dates)}")

        if self.daily_snapshots:
            final = self.daily_snapshots[-1]
            lines.append(f"Final equity: ${final.equity:,.2f}")
            pnl = final.equity - self.initial_cash
            lines.append(f"Total P&L: ${pnl:,.2f}")

        return "\n".join(lines)
