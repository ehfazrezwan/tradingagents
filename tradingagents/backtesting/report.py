import json
from pathlib import Path
from typing import Dict

from .results import BacktestResult


class BacktestReport:
    """Generates and saves backtest reports in various formats."""

    def __init__(self, result: BacktestResult):
        self.result = result

    def generate_text(self) -> str:
        """Generate a human-readable text report."""
        return self.result.summary()

    def generate_json(self) -> Dict:
        """Generate a JSON-serializable report dict."""
        r = self.result
        report = {
            "summary": {
                "ticker": r.ticker,
                "period": f"{r.start_date} to {r.end_date}",
                "initial_cash": r.initial_cash,
                "final_equity": r.daily_snapshots[-1].equity if r.daily_snapshots else r.initial_cash,
                "total_pnl": (r.daily_snapshots[-1].equity - r.initial_cash) if r.daily_snapshots else 0,
                "total_trades": len(r.trades),
                "failed_dates": len(r.failed_dates),
                "elapsed_seconds": r.elapsed_seconds,
            },
            "strategy_metrics": r.metrics,
            "benchmark_metrics": r.benchmark_metrics,
            "signals": r.signals,
        }
        return report

    def save(self, directory: str) -> None:
        """Save full report artifacts to a directory.

        Creates:
          - report.json (summary + metrics)
          - equity_curve.csv (daily snapshots)
          - full_result.json (complete result data)
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Summary report
        report_data = self.generate_json()
        with open(out_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Equity curve CSV
        self.result.to_csv(str(out_dir / "equity_curve.csv"))

        # Full result JSON
        self.result.to_json(str(out_dir / "full_result.json"))
