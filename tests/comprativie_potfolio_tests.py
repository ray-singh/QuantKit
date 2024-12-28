import pandas as pd
from typing import List
from QuantKit.data_fetching import fetch_data, fetch_company_info
import numpy as np
from QuantKit.Portfolio.Portfolio import Portfolio
from QuantKit.Portfolio.CPA import *
import pytest
from unittest.mock import MagicMock

class TestPortfolioAnalysis:
    @pytest.fixture
    def setup(self):
        portfolio1 = Portfolio(["AAPL", "MSFT"])
        portfolio2 = Portfolio(["GOOGL", "AMZN"])
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        return portfolio1, portfolio2, start_date, end_date

    def test_compare_returns(self, setup):
        portfolio1, portfolio2, start_date, end_date = setup
        fetch_data = MagicMock(return_value=pd.DataFrame({"Close": [100, 110, 120]}))
        result = compare_returns([portfolio1, portfolio2], start_date, end_date)
        assert not result.empty

    def test_compare_volatility(self, setup):
        portfolio1, portfolio2, start_date, end_date = setup
        fetch_data = MagicMock(return_value=pd.DataFrame({"Close": [100, 110, 120]}))
        result = compare_volatility([portfolio1, portfolio2], start_date, end_date)
        assert not result.empty

    def test_compare_sharpe_ratios(self, setup):
        portfolio1, portfolio2, start_date, end_date = setup
        fetch_data = MagicMock(return_value=pd.DataFrame({"Close": [100, 110, 120]}))
        result = compare_sharpe_ratios([portfolio1, portfolio2], start_date, end_date)
        assert not result.empty

    def test_compare_compositions(self, setup):
        portfolio1, portfolio2, _, _ = setup
        fetch_company_info = MagicMock(return_value={"Sector": "Technology"})
        result = compare_compositions([portfolio1, portfolio2])
        assert not result.empty