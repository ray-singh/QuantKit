import pytest
import pandas as pd
import numpy as np
from QuantKit.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_obv,
    calculate_stochastic_oscillator,
    calculate_vpt
)

@pytest.fixture
def sample_data():
    data = {
        'Close': [100, 102, 101, 105, 107, 110, 108, 111],
        'Volume': [1000, 1500, 1200, 1300, 1600, 1700, 1400, 1500],
        'High': [102, 103, 104, 106, 108, 111, 109, 113],
        'Low': [98, 99, 100, 103, 105, 109, 106, 110]
    }
    return pd.DataFrame(data)

# Test SMA
def test_calculate_sma(sample_data):
    result = calculate_sma(sample_data, window=3)
    expected = pd.Series([np.nan, np.nan, 101.0, 102.67, 104.33, 107.33, 108.33, 109.67], name='Close')
    pd.testing.assert_series_equal(result, expected, check_exact=False, atol=0.01)

# Test EMA
def test_calculate_ema(sample_data):
    result = calculate_ema(sample_data, window=3)
    assert len(result) == len(sample_data)

# Test RSI
def test_calculate_rsi(sample_data):
    result = calculate_rsi(sample_data, window=3)
    assert len(result) == len(sample_data)

# Test MACD
def test_calculate_macd(sample_data):
    result = calculate_macd(sample_data)
    assert 'MACD' in result.columns
    assert 'Signal Line' in result.columns
    assert len(result) == len(sample_data)

# Test Bollinger Bands
def test_calculate_bollinger_bands(sample_data):
    result = calculate_bollinger_bands(sample_data)
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
    assert isinstance(result, list)
