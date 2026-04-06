import logging
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Fetches historical close prices for P&L computation.

    Bulk-fetches the entire date range on first call, then serves
    individual prices from the cached DataFrame. This is separate
    from the agent's data tools to avoid any data leakage concerns.
    """

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def preload(self, ticker: str, start: str, end: str) -> None:
        """Bulk-fetch OHLCV data for the full backtest period.

        Fetches a few extra days before start and after end to ensure
        we have prices for edge trading days.
        """
        start_dt = pd.Timestamp(start) - pd.Timedelta(days=7)
        end_dt = pd.Timestamp(end) + pd.Timedelta(days=7)

        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(
                f"No price data returned for {ticker} between {start} and {end}"
            )

        # Flatten multi-level columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index).tz_localize(None)
        self._cache[ticker] = df
        logger.info(
            f"Preloaded {len(df)} days of price data for {ticker} "
            f"({df.index[0].date()} to {df.index[-1].date()})"
        )

    def get_close_price(self, ticker: str, date: str) -> Optional[float]:
        """Get the closing price for a ticker on a specific date.

        Returns None if no data available for that date.
        """
        if ticker not in self._cache:
            raise ValueError(
                f"No preloaded data for {ticker}. Call preload() first."
            )

        df = self._cache[ticker]
        dt = pd.Timestamp(date)

        if dt in df.index:
            return float(df.loc[dt, "Close"])

        # Try to find the nearest prior trading day (within 5 days)
        mask = df.index <= dt
        if mask.any():
            nearest = df.index[mask][-1]
            if (dt - nearest).days <= 5:
                return float(df.loc[nearest, "Close"])

        return None

    def get_trading_days(self, ticker: str, start: str, end: str) -> List[str]:
        """Return a list of trading day date strings within [start, end]."""
        if ticker not in self._cache:
            raise ValueError(
                f"No preloaded data for {ticker}. Call preload() first."
            )

        df = self._cache[ticker]
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        return [d.strftime("%Y-%m-%d") for d in df.index[mask]]

    def get_next_trading_day_price(
        self, ticker: str, date: str
    ) -> Optional[float]:
        """Get the close price on the next trading day after the given date.

        Useful for computing returns: decision on date D, price realized on D+1.
        """
        if ticker not in self._cache:
            raise ValueError(
                f"No preloaded data for {ticker}. Call preload() first."
            )

        df = self._cache[ticker]
        dt = pd.Timestamp(date)
        future = df.index[df.index > dt]
        if len(future) == 0:
            return None
        return float(df.loc[future[0], "Close"])
