"""
Portfolio optimization via PyPortfolioOpt.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting

TRADING_DAYS = 252

def annualize_returns(daily_returns: pd.DataFrame) -> pd.Series:
    mu_daily = daily_returns.mean()
    return (1 + mu_daily) ** TRADING_DAYS - 1

def get_mu_cov(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    prices: wide DataFrame of Adj Close with columns as tickers.
    """
    mu = mean_historical_return(prices, frequency=TRADING_DAYS)
    S  = CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    return mu, S

def efficient_frontier(mu: pd.Series, Sigma: pd.DataFrame):
    return EfficientFrontier(mu, Sigma)

def max_sharpe_portfolio(mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float = 0.0):
    ef = EfficientFrontier(mu, Sigma)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    w = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    return w, perf  # weights dict, (ret, vol, sharpe)

def min_vol_portfolio(mu: pd.Series, Sigma: pd.DataFrame):
    ef = EfficientFrontier(mu, Sigma)
    ef.min_volatility()
    w = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    return w, perf
