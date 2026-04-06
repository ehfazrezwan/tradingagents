import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Position:
    """Tracks shares held and average cost basis for a single ticker."""

    shares: float = 0.0
    avg_cost: float = 0.0

    @property
    def market_value(self) -> float:
        """Not useful without a price; use value_at(price) instead."""
        return 0.0

    def value_at(self, price: float) -> float:
        return self.shares * price


@dataclass
class Trade:
    """Record of a single executed trade."""

    date: str
    ticker: str
    action: str  # "BUY" or "SELL"
    shares: float
    price: float
    cost: float  # total transaction cost (fees + slippage)
    signal: str  # original signal that triggered this trade


@dataclass
class DailySnapshot:
    """End-of-day portfolio state."""

    date: str
    equity: float
    cash: float
    position_value: float
    signal: Optional[str] = None


class Portfolio:
    """Virtual portfolio that tracks cash, positions, and trade history."""

    def __init__(
        self,
        initial_cash: float,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ):
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[Trade] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps

    def _get_position(self, ticker: str) -> Position:
        if ticker not in self.positions:
            self.positions[ticker] = Position()
        return self.positions[ticker]

    def _compute_costs(self, shares: float, price: float) -> float:
        """Compute transaction cost + slippage for a trade."""
        notional = abs(shares) * price
        cost_rate = (self.transaction_cost_bps + self.slippage_bps) / 10_000
        return notional * cost_rate

    def execute_signal(
        self,
        ticker: str,
        signal: str,
        current_price: float,
        date: str,
        allocation_map: Dict[str, Optional[float]],
    ) -> Optional[Trade]:
        """
        Execute a position change based on a signal.

        Maps the signal to a target allocation %, computes the diff against
        current position, and executes the necessary buy or sell.

        Returns the Trade if one was executed, or None if no action taken.
        """
        target_alloc = allocation_map.get(signal)
        if target_alloc is None:
            # HOLD or unknown signal -> no action
            return None

        # Current portfolio equity at current prices
        total_equity = self.get_equity({ticker: current_price})

        # Target position value and shares
        target_value = total_equity * target_alloc
        pos = self._get_position(ticker)
        current_value = pos.value_at(current_price)
        diff_value = target_value - current_value

        if abs(diff_value) < current_price:
            # Less than 1 share difference, skip
            return None

        diff_shares = diff_value / current_price
        trade_cost = self._compute_costs(abs(diff_shares), current_price)

        if diff_shares > 0:
            # BUY
            shares_to_buy = math.floor(diff_shares)
            if shares_to_buy <= 0:
                return None
            total_outlay = shares_to_buy * current_price + trade_cost
            if total_outlay > self.cash:
                # Buy what we can afford
                shares_to_buy = math.floor(
                    (self.cash - trade_cost) / current_price
                )
                if shares_to_buy <= 0:
                    return None
                trade_cost = self._compute_costs(shares_to_buy, current_price)
                total_outlay = shares_to_buy * current_price + trade_cost

            # Update position with weighted average cost
            old_total = pos.shares * pos.avg_cost
            new_total = shares_to_buy * current_price
            pos.shares += shares_to_buy
            if pos.shares > 0:
                pos.avg_cost = (old_total + new_total) / pos.shares
            self.cash -= total_outlay

            trade = Trade(
                date=date,
                ticker=ticker,
                action="BUY",
                shares=shares_to_buy,
                price=current_price,
                cost=trade_cost,
                signal=signal,
            )
        else:
            # SELL
            shares_to_sell = min(math.floor(abs(diff_shares)), pos.shares)
            if shares_to_sell <= 0:
                return None
            trade_cost = self._compute_costs(shares_to_sell, current_price)
            proceeds = shares_to_sell * current_price - trade_cost
            pos.shares -= shares_to_sell
            if pos.shares == 0:
                pos.avg_cost = 0.0
            self.cash += proceeds

            trade = Trade(
                date=date,
                ticker=ticker,
                action="SELL",
                shares=shares_to_sell,
                price=current_price,
                cost=trade_cost,
                signal=signal,
            )

        self.trade_log.append(trade)
        return trade

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Total portfolio value: cash + sum of all position values."""
        position_value = sum(
            pos.value_at(prices.get(ticker, 0.0))
            for ticker, pos in self.positions.items()
        )
        return self.cash + position_value

    def get_position_value(self, prices: Dict[str, float]) -> float:
        """Total value of all positions."""
        return sum(
            pos.value_at(prices.get(ticker, 0.0))
            for ticker, pos in self.positions.items()
        )

    def snapshot(self, date: str, prices: Dict[str, float], signal: Optional[str] = None) -> DailySnapshot:
        """Record end-of-day portfolio state."""
        position_value = self.get_position_value(prices)
        snap = DailySnapshot(
            date=date,
            equity=self.cash + position_value,
            cash=self.cash,
            position_value=position_value,
            signal=signal,
        )
        self.daily_snapshots.append(snap)
        return snap

    def get_returns_series(self) -> pd.Series:
        """Compute daily returns from snapshots."""
        if len(self.daily_snapshots) < 2:
            return pd.Series(dtype=float)

        equities = pd.Series(
            [s.equity for s in self.daily_snapshots],
            index=pd.Index([s.date for s in self.daily_snapshots], name="date"),
        )
        return equities.pct_change().dropna()
