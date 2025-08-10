"""
Simple backtesting utilities.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple

TRADING_DAYS = 252

def compute_portfolio_returns(daily_returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    w = pd.Series(weights).reindex(daily_returns.columns).fillna(0.0)
    port_ret = daily_returns.mul(w, axis=1).sum(axis=1)
    port_ret.name = "portfolio_return"
    return port_ret

def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1

def sharpe_ratio(returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = returns - rf_daily
    mu = excess.mean() * TRADING_DAYS
    sigma = excess.std(ddof=0) * np.sqrt(TRADING_DAYS)
    if sigma == 0:
        return np.nan
    return float(mu / sigma)

def benchmark_60_40(daily_returns: pd.DataFrame) -> pd.Series:
    cols = daily_returns.columns
    if "SPY" not in cols or "BND" not in cols:
        raise ValueError("Daily returns must include SPY and BND for 60/40 benchmark.")
    weights = {"SPY": 0.60, "BND": 0.40}
    return compute_portfolio_returns(daily_returns[["SPY","BND"]], weights)
