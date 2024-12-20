# portfolio.py
import pandas as pd
import numpy as np
from data_fetching import fetch_data, fetch_company_info, fetch_live_price
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')

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
        Plot the cumulative returns of the portfolio.
        """
        weighted_returns = self.returns.dot(list(self.weights.values()))
        cumulative_returns = (1 + weighted_returns).cumprod()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, label='Portfolio Cumulative Returns', color='blue')
        plt.title('Portfolio Cumulative Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_stock_allocation(self):
        """
        Plot a pie chart showing stock allocation in the portfolio.
        """
        plt.figure(figsize=(8, 8))
        plt.pie(self.weights.values(), labels=self.weights.keys(), autopct='%1.1f%%', startangle=140,
                colors=sns.color_palette('pastel'))
        plt.title("Portfolio Stock Allocation")
        plt.show()

    def plot_individual_stock_performance(self):
        """
        Plot the cumulative returns of individual stocks in the portfolio.
        """
        plt.figure(figsize=(10, 6))
        for stock, data in self.stocks.items():
            stock_returns = (1 + data['Close'].pct_change()).cumprod()
            plt.plot(stock_returns.index, stock_returns, label=stock)
        plt.title("Individual Stock Performance")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_risk_vs_return(self):
        """
        Plot risk (volatility) vs return for all stocks in the portfolio.
        """
        returns = self.returns.mean() * 252  # Annualized returns
        volatilities = self.returns.std() * np.sqrt(252)  # Annualized volatilities
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=volatilities, y=returns, s=100, color='red')
        for stock in self.stock_symbols:
            plt.text(volatilities[stock], returns[stock], stock, fontsize=10)
        plt.title("Risk vs Return")
        plt.xlabel("Volatility (Risk)")
        plt.ylabel("Annualized Return")
        plt.grid()
        plt.show()

    def compare_with_benchmark(self, benchmark_symbol):
        """
        Compare the portfolio's cumulative returns with a benchmark.

        Parameters:
        benchmark_symbol (str): Stock symbol for the benchmark (e.g., 'SPY').
        """
        benchmark_data = fetch_data(benchmark_symbol, self.start_date, self.end_date)['Close'].pct_change()
        portfolio_weighted_returns = self.returns.dot(list(self.weights.values()))
        portfolio_cumulative = (1 + portfolio_weighted_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_data).cumprod()

        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_cumulative.index, portfolio_cumulative, label='Portfolio', color='blue')
        plt.plot(benchmark_cumulative.index, benchmark_cumulative, label=benchmark_symbol, color='green')
        plt.title("Portfolio vs Benchmark Performance")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid()
        plt.show()
