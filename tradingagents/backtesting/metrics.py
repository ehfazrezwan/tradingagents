import math
from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252,
) -> Dict:
    """Compute standard performance metrics from a daily returns series.

    Args:
        daily_returns: Series of daily percentage returns (e.g., 0.01 = 1%).
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino computation.
        trading_days_per_year: Number of trading days per year.

    Returns:
        Dictionary of performance metrics.
    """
    if len(daily_returns) < 2:
        return _empty_metrics()

    returns = daily_returns.values.astype(float)
    n_days = len(returns)
    daily_rf = risk_free_rate / trading_days_per_year

    # Total and annualized return
    cumulative = np.prod(1 + returns) - 1
    years = n_days / trading_days_per_year
    annualized_return = (1 + cumulative) ** (1 / years) - 1 if years > 0 else 0.0

    # Volatility
    vol_daily = np.std(returns, ddof=1) if n_days > 1 else 0.0
    vol_annual = vol_daily * math.sqrt(trading_days_per_year)

    # Sharpe ratio
    excess_returns = returns - daily_rf
    sharpe = (
        np.mean(excess_returns) / np.std(excess_returns, ddof=1) * math.sqrt(trading_days_per_year)
        if np.std(excess_returns, ddof=1) > 0
        else 0.0
    )

    # Sortino ratio (downside deviation only)
    downside = excess_returns[excess_returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    sortino = (
        np.mean(excess_returns) / downside_std * math.sqrt(trading_days_per_year)
        if downside_std > 0
        else 0.0
    )

    # Max drawdown
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))  # negative number

    # Max drawdown duration (in trading days)
    max_dd_duration = _max_drawdown_duration(cum_returns)

    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Win rate
    winning_days = np.sum(returns > 0)
    losing_days = np.sum(returns < 0)
    total_active = winning_days + losing_days
    win_rate = float(winning_days / total_active) if total_active > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average win / loss
    avg_win = float(np.mean(returns[returns > 0])) if winning_days > 0 else 0.0
    avg_loss = float(np.mean(returns[returns < 0])) if losing_days > 0 else 0.0

    return {
        "total_return": float(cumulative),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_duration_days": max_dd_duration,
        "calmar_ratio": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_trading_days": n_days,
        "volatility_annualized": float(vol_annual),
    }


def compute_benchmark_returns(
    price_series: pd.Series,
) -> pd.Series:
    """Compute buy-and-hold daily returns from a price series.

    Args:
        price_series: Series of daily close prices indexed by date.

    Returns:
        Series of daily percentage returns.
    """
    return price_series.pct_change().dropna()


def _max_drawdown_duration(cum_returns: np.ndarray) -> int:
    """Compute the longest drawdown duration in trading days."""
    running_max = np.maximum.accumulate(cum_returns)
    in_drawdown = cum_returns < running_max

    max_duration = 0
    current_duration = 0
    for dd in in_drawdown:
        if dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def _empty_metrics() -> Dict:
    """Return a metrics dict with zero/default values."""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration_days": 0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "total_trading_days": 0,
        "volatility_annualized": 0.0,
    }
