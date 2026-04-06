"""Simple rule-based benchmark strategies for comparison.

Each strategy takes a price DataFrame and returns a list of DailySnapshots
with the same format as the Portfolio-based backtester, enabling direct
metric comparison via compute_metrics().
"""

from typing import List

import numpy as np
import pandas as pd

from .portfolio import DailySnapshot


def buy_and_hold(
    prices: pd.Series,
    initial_cash: float = 100_000.0,
) -> List[DailySnapshot]:
    """Buy-and-hold benchmark: invest all cash on day 1, hold forever.

    Args:
        prices: Series of daily close prices indexed by date string.
        initial_cash: Starting cash amount.

    Returns:
        List of DailySnapshot representing the daily equity curve.
    """
    if prices.empty:
        return []

    first_price = prices.iloc[0]
    shares = int(initial_cash // first_price)
    remaining_cash = initial_cash - shares * first_price

    snapshots = []
    for date, price in prices.items():
        position_value = shares * price
        equity = remaining_cash + position_value
        snapshots.append(
            DailySnapshot(
                date=str(date),
                equity=equity,
                cash=remaining_cash,
                position_value=position_value,
                signal="HOLD",
            )
        )

    return snapshots


def momentum_strategy(
    prices: pd.Series,
    lookback: int = 20,
    initial_cash: float = 100_000.0,
    transaction_cost_bps: float = 10.0,
) -> List[DailySnapshot]:
    """Simple momentum strategy: buy if N-day return > 0, else sell/stay out.

    Args:
        prices: Series of daily close prices indexed by date string.
        lookback: Number of days for momentum lookback.
        initial_cash: Starting cash amount.
        transaction_cost_bps: Transaction cost in basis points.

    Returns:
        List of DailySnapshot.
    """
    if len(prices) <= lookback:
        return buy_and_hold(prices, initial_cash)

    cost_rate = transaction_cost_bps / 10_000
    cash = initial_cash
    shares = 0
    snapshots = []

    prices_arr = prices.values.astype(float)
    dates = [str(d) for d in prices.index]

    for i, (date, price) in enumerate(zip(dates, prices_arr)):
        if i >= lookback:
            momentum_return = (price - prices_arr[i - lookback]) / prices_arr[i - lookback]

            if momentum_return > 0 and shares == 0:
                # Buy
                affordable = int((cash * (1 - cost_rate)) // price)
                if affordable > 0:
                    cost = affordable * price * cost_rate
                    cash -= affordable * price + cost
                    shares = affordable

            elif momentum_return <= 0 and shares > 0:
                # Sell
                cost = shares * price * cost_rate
                cash += shares * price - cost
                shares = 0

        position_value = shares * price
        equity = cash + position_value
        signal = "BUY" if shares > 0 else "SELL"
        snapshots.append(
            DailySnapshot(
                date=date,
                equity=equity,
                cash=cash,
                position_value=position_value,
                signal=signal,
            )
        )

    return snapshots


def mean_reversion_strategy(
    prices: pd.Series,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    initial_cash: float = 100_000.0,
    transaction_cost_bps: float = 10.0,
) -> List[DailySnapshot]:
    """RSI-based mean reversion: buy when oversold, sell when overbought.

    Args:
        prices: Series of daily close prices indexed by date string.
        rsi_period: RSI calculation period.
        oversold: RSI threshold to buy.
        overbought: RSI threshold to sell.
        initial_cash: Starting cash amount.
        transaction_cost_bps: Transaction cost in basis points.

    Returns:
        List of DailySnapshot.
    """
    if len(prices) <= rsi_period + 1:
        return buy_and_hold(prices, initial_cash)

    cost_rate = transaction_cost_bps / 10_000
    cash = initial_cash
    shares = 0
    snapshots = []

    prices_arr = prices.values.astype(float)
    dates = [str(d) for d in prices.index]

    # Compute RSI
    rsi_values = _compute_rsi(prices_arr, rsi_period)

    for i, (date, price) in enumerate(zip(dates, prices_arr)):
        if i >= rsi_period and rsi_values[i] is not None:
            rsi = rsi_values[i]

            if rsi < oversold and shares == 0:
                # Buy on oversold
                affordable = int((cash * (1 - cost_rate)) // price)
                if affordable > 0:
                    cost = affordable * price * cost_rate
                    cash -= affordable * price + cost
                    shares = affordable

            elif rsi > overbought and shares > 0:
                # Sell on overbought
                cost = shares * price * cost_rate
                cash += shares * price - cost
                shares = 0

        position_value = shares * price
        equity = cash + position_value
        signal = "BUY" if shares > 0 else "SELL" if i >= rsi_period else "HOLD"
        snapshots.append(
            DailySnapshot(
                date=date,
                equity=equity,
                cash=cash,
                position_value=position_value,
                signal=signal,
            )
        )

    return snapshots


def _compute_rsi(prices: np.ndarray, period: int) -> list:
    """Compute RSI values for a price array.

    Returns a list of RSI values (None for the first `period` entries).
    """
    deltas = np.diff(prices)
    rsi_list = [None] * (period)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi_list.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_list.append(100.0 - 100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_list.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_list.append(100.0 - 100.0 / (1.0 + rs))

    return rsi_list
