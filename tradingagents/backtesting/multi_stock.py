"""Multi-stock backtesting: run independent backtests across multiple tickers."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import BacktestConfig
from .results import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class MultiStockResult:
    """Results from backtesting across multiple tickers."""

    results: Dict[str, BacktestResult] = field(default_factory=dict)
    failed_tickers: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary across all tickers."""
        lines = ["=" * 60, "MULTI-STOCK BACKTEST RESULTS", "=" * 60, ""]

        for ticker, result in sorted(self.results.items()):
            metrics = result.metrics
            total_ret = metrics.get("total_return", 0.0)
            sharpe = metrics.get("sharpe_ratio", 0.0)
            max_dd = metrics.get("max_drawdown", 0.0)
            lines.append(
                f"  {ticker:8s}  Return: {total_ret:+7.2%}  "
                f"Sharpe: {sharpe:6.2f}  MaxDD: {max_dd:+7.2%}"
            )

        if self.failed_tickers:
            lines.append("")
            lines.append(f"  Failed: {', '.join(self.failed_tickers)}")

        # Aggregate stats
        if self.results:
            all_returns = [
                r.metrics.get("total_return", 0.0) for r in self.results.values()
            ]
            avg_return = sum(all_returns) / len(all_returns)
            lines.append("")
            lines.append(f"  Average Return: {avg_return:+.2%} across {len(self.results)} tickers")

        return "\n".join(lines)


class MultiStockBacktester:
    """Run independent backtests for multiple tickers.

    Each ticker gets its own TradingAgentsGraph instance with independent
    memory systems. Can be parallelized since tickers don't share state.
    """

    def __init__(
        self,
        tickers: List[str],
        base_config: BacktestConfig,
        max_workers: int = 1,
    ):
        """
        Args:
            tickers: List of ticker symbols to backtest.
            base_config: Base configuration (ticker field will be overridden).
            max_workers: Number of parallel workers. Set >1 for parallel execution.
                Note: parallel execution uses more memory and may hit API rate limits.
        """
        self.tickers = tickers
        self.base_config = base_config
        self.max_workers = max_workers

    def run(self, progress_callback=None) -> MultiStockResult:
        """Run backtests for all tickers.

        Args:
            progress_callback: Optional callable(ticker, status) where status
                is "started", "completed", or "failed".

        Returns:
            MultiStockResult with per-ticker results.
        """
        from .runner import BacktestRunner

        multi_result = MultiStockResult()

        if self.max_workers <= 1:
            # Sequential execution
            for ticker in self.tickers:
                self._run_single(
                    ticker, BacktestRunner, multi_result, progress_callback
                )
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_single_return,
                        ticker,
                        BacktestRunner,
                    ): ticker
                    for ticker in self.tickers
                }
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            multi_result.results[ticker] = result
                            if progress_callback:
                                progress_callback(ticker, "completed")
                        else:
                            multi_result.failed_tickers.append(ticker)
                            if progress_callback:
                                progress_callback(ticker, "failed")
                    except Exception as e:
                        logger.error(f"Ticker {ticker} failed: {e}")
                        multi_result.failed_tickers.append(ticker)
                        if progress_callback:
                            progress_callback(ticker, "failed")

        return multi_result

    def _make_ticker_config(self, ticker: str) -> BacktestConfig:
        """Create a config for a specific ticker, copying base settings."""
        return BacktestConfig(
            ticker=ticker,
            start_date=self.base_config.start_date,
            end_date=self.base_config.end_date,
            initial_cash=self.base_config.initial_cash,
            signal_allocation=self.base_config.signal_allocation,
            transaction_cost_bps=self.base_config.transaction_cost_bps,
            slippage_bps=self.base_config.slippage_bps,
            enable_reflection=self.base_config.enable_reflection,
            reflection_frequency=self.base_config.reflection_frequency,
            cache_analyst_outputs=self.base_config.cache_analyst_outputs,
            cache_dir=self.base_config.cache_dir,
            graph_config=self.base_config.graph_config.copy(),
            selected_analysts=list(self.base_config.selected_analysts),
            results_dir=self.base_config.results_dir,
            max_retries=self.base_config.max_retries,
        )

    def _run_single(
        self, ticker, runner_cls, multi_result, progress_callback=None
    ):
        """Run a single ticker backtest and store in multi_result."""
        if progress_callback:
            progress_callback(ticker, "started")

        try:
            config = self._make_ticker_config(ticker)
            runner = runner_cls(config)
            result = runner.run()
            multi_result.results[ticker] = result
            if progress_callback:
                progress_callback(ticker, "completed")
        except Exception as e:
            logger.error(f"Backtest failed for {ticker}: {e}")
            multi_result.failed_tickers.append(ticker)
            if progress_callback:
                progress_callback(ticker, "failed")

    def _run_single_return(
        self, ticker, runner_cls
    ) -> Optional[BacktestResult]:
        """Run a single ticker and return the result (for parallel execution)."""
        try:
            config = self._make_ticker_config(ticker)
            runner = runner_cls(config)
            return runner.run()
        except Exception as e:
            logger.error(f"Backtest failed for {ticker}: {e}")
            return None
