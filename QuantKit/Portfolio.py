# portfolio.py
import pandas as pd
import numpy as np
from data_fetching import fetch_data, fetch_company_info, fetch_live_price
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt


class Portfolio:
    def __init__(self, stock_symbols=None, start_date='2020-01-01', end_date='2024-01-01', weights=None):
        """
        Initialize the Portfolio with stock symbols and fetch data from Yahoo Finance.

        Parameters:
        stock_symbols (list): A list of stock symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date for the data (default is '2020-01-01').
        end_date (str): End date for the data (default is '2024-01-01').
        weights (dict or list, optional): A dictionary or list of portfolio weights for each stock.
        """
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.stocks = self.fetch_stocks_data()
        self.weights = weights if weights else {stock: 1 / len(stock_symbols) for stock in stock_symbols}
        self.returns = self.calculate_returns()

    def fetch_stocks_data(self):
        """
        Fetch stock price data using the fetch_data function from data_fetching.py.

        Returns:
        dict: A dictionary where keys are stock symbols and values are pandas DataFrames with stock price data.
        """
        stocks_data = {}
        for symbol in self.stock_symbols:
            data = fetch_data(symbol, self.start_date, self.end_date)
            if not data.empty:
                stocks_data[symbol] = data
        return stocks_data

    def calculate_returns(self):
        """
        Calculate daily returns for each stock in the portfolio.

        Returns:
        pd.DataFrame: DataFrame of daily returns for each stock in the portfolio.
        """
        returns = pd.DataFrame()
        for stock, data in self.stocks.items():
            returns[stock] = data['Close'].pct_change().dropna()
        return returns

    def calculate_portfolio_return(self):
        """
        Calculate the portfolio's return based on weights and stock returns.

        Returns:
        float: The annualized portfolio return.
        """
        weighted_returns = self.returns.dot(list(self.weights.values()))
        portfolio_return = weighted_returns.mean() * 252  # Annualize return
        return portfolio_return

    def calculate_volatility(self):
        """
        Calculate the portfolio's volatility (standard deviation).

        Returns:
        float: The annualized portfolio volatility.
        """
        covariance_matrix = self.returns.cov() * 252  # Annualized covariance
        portfolio_variance = np.dot(list(self.weights.values()), np.dot(covariance_matrix, list(self.weights.values())))
        portfolio_volatility = np.sqrt(portfolio_variance)
        return portfolio_volatility


    def add_stock(self, stock_symbol, weight=None):
        """
        Add a new stock to the portfolio and fetch its data.

        Parameters:
        stock_symbol (str): The stock symbol to add (e.g., 'GOOG').
        weight (float, optional): The weight of the new stock in the portfolio. If None, an equal weight is assigned.
        """
        if stock_symbol not in self.stocks:
            data = fetch_data(stock_symbol, self.start_date, self.end_date)
            if not data.empty:
                self.stocks[stock_symbol] = data
                self.weights[stock_symbol] = weight if weight else 1 / len(self.stocks)
                self.returns = self.calculate_returns()

    def remove_stock(self, stock_symbol):
        """
        Remove a stock from the portfolio.

        Parameters:
        stock_symbol (str): The stock symbol to be removed.
        """
        if stock_symbol in self.stocks:
            del self.stocks[stock_symbol]
            del self.weights[stock_symbol]
            self.returns = self.calculate_returns()
        else:
            print(f"Stock {stock_symbol} not found in portfolio.")

    def get_company_info(self, stock_symbol):
        """
        Get basic company information such as sector, industry, and description.

        Parameters:
        stock_symbol (str): The stock symbol to get information for.

        Returns:
        dict: Company information dictionary.
        """
        return fetch_company_info(stock_symbol)

    def get_live_price(self, stock_symbol):
        """
        Get the live price of a stock.

        Parameters:
        stock_symbol (str): The stock symbol to get the live price for.

        Returns:
        float: The current stock price.
        """
        return fetch_live_price(stock_symbol)

    def plot_portfolio_performance(self):
        """
        Plots the cumulative returns of the portfolio over time.
        """
        plt.figure(figsize=(10, 6))
        cumulative_returns = (1 + self.portfolio_data.pct_change()).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.sum(axis=1), label='Portfolio Cumulative Returns',
                 color='blue')
        plt.title("Portfolio Performance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_stock_allocation(self, weights: List[float]):
        """
        Plots a pie chart showing stock allocation in the portfolio.

        Args:
            weights (List[float]): List of weights for each stock in the portfolio.
        """
        plt.figure(figsize=(8, 8))
        labels = self.tickers
        plt.pie(weights, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title("Portfolio Stock Allocation")
        plt.show()

    def compare_portfolio_returns(self, portfolio_returns: Dict[str, float]):
        """
        Plots a bar chart comparing the returns of multiple portfolios.

        Args:
            portfolio_returns (Dict[str, float]): Dictionary of portfolio names and their annualized returns.
        """
        plt.figure(figsize=(10, 6))
        names = list(portfolio_returns.keys())
        returns = list(portfolio_returns.values())
        sns.barplot(x=names, y=returns, palette="viridis")
        plt.title("Comparative Portfolio Returns")
        plt.ylabel("Annualized Return")
        plt.xlabel("Portfolio")
        plt.show()

    def plot_risk_vs_return(self, portfolio_stats: Dict[str, tuple]):
        """
        Plots a scatter plot comparing risk (volatility) and return across portfolios or stocks.

        Args:
            portfolio_stats (Dict[str, tuple]): Dictionary of portfolio/stock names with (return, risk) values.
        """
        plt.figure(figsize=(10, 6))
        for name, stats in portfolio_stats.items():
            plt.scatter(stats[1], stats[0], label=name, s=100)  # stats[0] = return, stats[1] = risk
            plt.text(stats[1], stats[0], name, fontsize=10)

        plt.title("Risk vs. Return")
        plt.xlabel("Volatility (Risk)")
        plt.ylabel("Annualized Return")
        plt.legend()
        plt.grid()
        plt.show()
