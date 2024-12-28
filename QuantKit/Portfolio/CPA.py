import pandas as pd
from typing import List
from QuantKit.data_fetching import fetch_data, fetch_company_info
import numpy as np
from QuantKit.Portfolio.Portfolio import Portfolio

# comparitive portfolio analysis

def compare_returns(portfolios: List[Portfolio], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Compare cumulative returns of multiple portfolios over a specific time period.

    Args:
        portfolios (List[Portfolio]): List of Portfolio objects.
        start_date (str): Start date for analysis (YYYY-MM-DD).
        end_date (str): End date for analysis (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame with cumulative returns for each portfolio.
    """
    results = {}
    for portfolio in portfolios:
        cumulative_return = 0
        for ticker in portfolio.stock_symbols:
            data = fetch_data(ticker, start_date, end_date)
            if not data.empty:
                daily_returns = data['Close'].pct_change().dropna()
                cumulative_return += daily_returns.add(1).prod() - 1
        results[portfolio] = cumulative_return
    return pd.DataFrame(results.items(), columns=["Portfolio", "Cumulative Return"])


def compare_volatility(portfolios: List[Portfolio], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Compare the volatility (standard deviation of returns) of multiple portfolios.

    Args:
        portfolios (List[Portfolio]): List of Portfolio objects.
        start_date (str): Start date for analysis (YYYY-MM-DD).
        end_date (str): End date for analysis (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame with volatilities for each portfolio.
    """
    results = {}
    for portfolio in portfolios:
        total_volatility = 0
        for ticker in portfolio.stock_symbols:
            data = fetch_data(ticker, start_date, end_date)
            if not data.empty:
                daily_returns = data['Close'].pct_change().dropna()
                total_volatility += daily_returns.std()
        results[portfolio] = total_volatility
    return pd.DataFrame(results.items(), columns=["Portfolio", "Volatility"])


def compare_sharpe_ratios(portfolios: List[Portfolio], start_date: str, end_date: str, risk_free_rate=0.01) -> pd.DataFrame:
    """
    Compare Sharpe ratios of multiple portfolios.

    Args:
        portfolios (List[Portfolio]): List of Portfolio objects.
        start_date (str): Start date for analysis (YYYY-MM-DD).
        end_date (str): End date for analysis (YYYY-MM-DD).
        risk_free_rate (float): Risk-free rate (default 0.01).

    Returns:
        pd.DataFrame: DataFrame with Sharpe ratios for each portfolio.
    """
    results = {}
    for portfolio in portfolios:
        portfolio_return = 0
        portfolio_volatility = 0
        for ticker in portfolio.stock_symbols:
            data = fetch_data(ticker, start_date, end_date)
            if not data.empty:
                daily_returns = data['Close'].pct_change().dropna()
                portfolio_return += daily_returns.mean()
                portfolio_volatility += daily_returns.std()
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility else 0
        results[portfolio] = sharpe_ratio
    return pd.DataFrame(results.items(), columns=["Portfolio", "Sharpe Ratio"])


def compare_compositions(portfolios: List[Portfolio]) -> pd.DataFrame:
    """
    Compare the composition (sector/industry distribution) of multiple portfolios.

    Args:
        portfolios (List[Portfolio]): List of Portfolio objects.

    Returns:
        pd.DataFrame: DataFrame showing sector/industry allocation for each portfolio.
    """
    composition_results = {}
    for portfolio in portfolios:
        sector_counts = {}
        for ticker in portfolio.stock_symbols:
            info = fetch_company_info(ticker)
            if info:
                sector = info.get("Sector", "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        composition_results[portfolio] = sector_counts
    return pd.DataFrame(composition_results).fillna(0)
