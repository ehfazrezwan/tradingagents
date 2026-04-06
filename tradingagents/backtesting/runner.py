import logging
import time
from typing import Optional

import pandas as pd

from tradingagents.graph.trading_graph import TradingAgentsGraph

from .cache import AnalystCache
from .config import BacktestConfig
from .metrics import compute_benchmark_returns, compute_metrics
from .portfolio import Portfolio
from .price_fetcher import PriceFetcher
from .report import BacktestReport
from .results import BacktestResult
from .signal_mapper import SignalMapper

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Orchestrates a full backtest over a date range.

    Wraps TradingAgentsGraph.propagate() in a day-by-day loop with
    portfolio tracking, caching, reflection, and metrics computation.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.price_fetcher = PriceFetcher()
        self.signal_mapper = SignalMapper(config.signal_allocation)
        self.cache: Optional[AnalystCache] = None
        self._config_hash: str = ""

        if config.cache_analyst_outputs:
            self.cache = AnalystCache(config.cache_dir)
            self._config_hash = AnalystCache.config_hash(config.graph_config)

    def run(self, progress_callback=None) -> BacktestResult:
        """Execute the full backtest.

        Args:
            progress_callback: Optional callable(day_index, total_days, date, signal)
                for progress reporting (e.g., CLI progress bar).

        Returns:
            BacktestResult with all metrics, trades, and snapshots.
        """
        start_time = time.time()
        ticker = self.config.ticker

        # 1. Preload prices
        logger.info(f"Preloading price data for {ticker}...")
        self.price_fetcher.preload(ticker, self.config.start_date, self.config.end_date)
        trading_days = self.price_fetcher.get_trading_days(
            ticker, self.config.start_date, self.config.end_date
        )

        if not trading_days:
            raise ValueError(
                f"No trading days found for {ticker} between "
                f"{self.config.start_date} and {self.config.end_date}"
            )

        logger.info(f"Running backtest: {len(trading_days)} trading days")

        # 2. Initialize graph and portfolio
        graph = TradingAgentsGraph(
            selected_analysts=self.config.selected_analysts,
            config=self.config.graph_config,
        )
        portfolio = Portfolio(
            initial_cash=self.config.initial_cash,
            transaction_cost_bps=self.config.transaction_cost_bps,
            slippage_bps=self.config.slippage_bps,
        )

        # 3. Result tracking
        result = BacktestResult(
            ticker=ticker,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_cash=self.config.initial_cash,
        )

        # 4. Day-by-day loop
        for day_idx, date in enumerate(trading_days):
            price = self.price_fetcher.get_close_price(ticker, date)
            if price is None:
                logger.warning(f"No price for {date}, skipping")
                result.failed_dates.append(date)
                continue

            # Get signal (from cache or fresh propagation)
            signal = self._get_signal(graph, ticker, date, result)

            # Execute trade
            trade = portfolio.execute_signal(
                ticker=ticker,
                signal=signal,
                current_price=price,
                date=date,
                allocation_map=self.config.signal_allocation,
            )
            if trade:
                result.trades.append(trade)

            # Record snapshot
            portfolio.snapshot(date, {ticker: price}, signal=signal)
            result.signals.append((date, signal))

            # Periodic reflection
            if self.config.enable_reflection and self._should_reflect(day_idx):
                self._do_reflection(graph, portfolio, ticker)

            # Progress callback
            if progress_callback:
                progress_callback(day_idx, len(trading_days), date, signal)

            logger.info(
                f"[{day_idx + 1}/{len(trading_days)}] {date}: "
                f"signal={signal}, equity=${portfolio.get_equity({ticker: price}):,.2f}"
            )

        # 5. Compute metrics
        result.daily_snapshots = portfolio.daily_snapshots
        daily_returns = portfolio.get_returns_series()
        result.metrics = compute_metrics(daily_returns)

        # Benchmark (buy-and-hold)
        benchmark_prices = pd.Series(
            [s.equity for s in portfolio.daily_snapshots],
            index=[s.date for s in portfolio.daily_snapshots],
        )
        # For benchmark, compute using raw price returns
        bh_prices = pd.Series(
            {
                date: self.price_fetcher.get_close_price(ticker, date)
                for date in trading_days
                if self.price_fetcher.get_close_price(ticker, date) is not None
            }
        )
        if len(bh_prices) > 1:
            bh_returns = compute_benchmark_returns(bh_prices)
            result.benchmark_metrics = compute_metrics(bh_returns)

        result.elapsed_seconds = time.time() - start_time

        # 6. Save results
        self._save_results(result)

        return result

    def _get_signal(
        self,
        graph: TradingAgentsGraph,
        ticker: str,
        date: str,
        result: BacktestResult,
    ) -> str:
        """Get signal for a date, using cache if available."""
        # Check cache
        if self.cache:
            cached = self.cache.get(ticker, date, self._config_hash)
            if cached is not None:
                _, signal = cached
                logger.info(f"Cache hit for {date}: {signal}")
                return signal

        # Fresh propagation with retries
        signal = "HOLD"
        for attempt in range(self.config.max_retries + 1):
            try:
                state, signal = graph.propagate(ticker, date)

                # Cache the result
                if self.cache:
                    self.cache.put(ticker, date, self._config_hash, state, signal)

                return self.signal_mapper.normalize_signal(signal)
            except Exception as e:
                logger.warning(
                    f"Propagation failed for {date} (attempt {attempt + 1}): {e}"
                )
                if attempt == self.config.max_retries:
                    logger.error(f"All retries exhausted for {date}, defaulting to HOLD")
                    result.failed_dates.append(date)

        return "HOLD"

    def _should_reflect(self, day_index: int) -> bool:
        """Check if we should run reflection at this day index."""
        return (
            day_index > 0
            and day_index % self.config.reflection_frequency == 0
        )

    def _do_reflection(
        self,
        graph: TradingAgentsGraph,
        portfolio: Portfolio,
        ticker: str,
    ) -> None:
        """Run reflect_and_remember with recent returns."""
        returns = portfolio.get_returns_series()
        if len(returns) < 2:
            return

        # Use the most recent return window for reflection
        recent_returns = returns.iloc[-self.config.reflection_frequency :]
        avg_return = float(recent_returns.mean())

        try:
            graph.reflect_and_remember(avg_return)
            logger.info(f"Reflected with avg return: {avg_return:.4f}")
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")

    def _save_results(self, result: BacktestResult) -> None:
        """Save backtest results to disk."""
        report = BacktestReport(result)
        out_dir = f"{self.config.results_dir}/{self.config.ticker}_{self.config.start_date}_{self.config.end_date}"
        report.save(out_dir)
        logger.info(f"Results saved to {out_dir}/")
