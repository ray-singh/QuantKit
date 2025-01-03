import pytest
import pandas as pd
import numpy as np
from Stockify.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_obv,
    calculate_stochastic_oscillator,
    calculate_vpt
)
from QuantKit.data_fetching import fetch_data
@pytest.fixture
def sample_data():
    data = fetch_data(ticker='AAPL', start_date='2023-01-01', end_date='2023-12-31')
    return data

# Test SMA
def test_calculate_sma(sample_data):
    result = calculate_sma(sample_data, window=20)
    assert len(result) == len(sample_data)

# Test EMA
def test_calculate_ema(sample_data):
    result = calculate_ema(sample_data, window=20)
    assert len(result) == len(sample_data)

# Test RSI
def test_calculate_rsi(sample_data):
    result = calculate_rsi(sample_data, window=14)
    assert len(result) == len(sample_data)

# Test MACD
def test_calculate_macd(sample_data):
    result = calculate_macd(sample_data, ticker='AAPl')
    assert 'MACD' in result.columns
    assert 'Signal Line' in result.columns
    assert len(result) == len(sample_data)

# Test Bollinger Bands
def test_calculate_bollinger_bands(sample_data):
    result = calculate_bollinger_bands(sample_data, ticker='AAPL')
    assert 'Upper Band' in result.columns
    assert 'Middle Band (SMA)' in result.columns
    assert 'Lower Band' in result.columns
    assert len(result) == len(sample_data)

# Test OBV
def test_calculate_obv(sample_data):
    result = calculate_obv(sample_data)
    assert len(result) == len(sample_data)

# Test Stochastic Oscillator
def test_calculate_stochastic_oscillator(sample_data):
    result = calculate_stochastic_oscillator(sample_data)
    assert '%K' in result.columns
    assert '%D' in result.columns
    assert len(result) == len(sample_data)

# Test VPT
def test_calculate_vpt(sample_data):
    result = calculate_vpt(sample_data)
    assert len(result) == len(sample_data)
    assert isinstance(result, pd.Series)
