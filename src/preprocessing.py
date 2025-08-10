"""
Preprocessing, feature engineering, and diagnostics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from statsmodels.tsa.stattools import adfuller

TRADING_DAYS = 252

def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def compute_returns(df: pd.DataFrame, col: str = "Adj Close") -> pd.Series:
    df = to_datetime_index(df)
    ret = df[col].pct_change().dropna()
    ret.name = "returns"
    return ret

def compute_log_returns(df: pd.DataFrame, col: str = "Adj Close") -> pd.Series:
    df = to_datetime_index(df)
    r = np.log(df[col]).diff().dropna()
    r.name = "log_returns"
    return r

def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std()

def adf_test(series: pd.Series) -> Dict[str, float]:
    """
    Returns dict with statistic, pvalue, and critical values.
    """
    series = series.dropna().astype(float)
    res = adfuller(series, autolag="AIC")
    stat, pvalue, usedlag, nobs, crit, icbest = res
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "crit_1%": float(crit["1%"]),
        "crit_5%": float(crit["5%"]),
        "crit_10%": float(crit["10%"]),
        "icbest": float(icbest),
    }

def value_at_risk(returns: pd.Series, alpha: float = 0.95) -> float:
    """
    Historical VaR at given confidence (one-day). Positive value indicates loss magnitude.
    """
    q = returns.quantile(1 - alpha)
    return abs(float(q))

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, trading_days: int = TRADING_DAYS) -> float:
    """
    Annualized Sharpe using daily returns.
    rf is daily risk-free rate (set 0 if unavailable).
    """
    excess = returns - rf
    mu = excess.mean() * trading_days
    sigma = excess.std(ddof=0) * np.sqrt(trading_days)
    if sigma == 0:
        return np.nan
    return float(mu / sigma)
