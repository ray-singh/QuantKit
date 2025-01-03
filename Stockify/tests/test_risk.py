import pytest
import pandas as pd
import numpy as np
from Stockify.risk import value_at_risk, conditional_value_at_risk, maximum_drawdown

# Sample data for testing
returns = pd.Series([-0.02, 0.03, -0.04, 0.01, -0.01, 0.02, -0.03, 0.04, -0.02])
prices = pd.Series([100, 105, 102, 106, 104, 108, 103, 109, 105])

# Test value_at_risk
@pytest.mark.parametrize("confidence_level, expected", [(0.95, -0.03), (0.99, -0.04)])
def test_value_at_risk(confidence_level, expected):
    result = value_at_risk(returns, confidence_level)
    assert np.isclose(result, expected, atol=1e-2)


def test_value_at_risk_invalid_input():
    with pytest.raises(ValueError):
        value_at_risk([1, 2, 3], 0.95)  # Invalid input type

    with pytest.raises(ValueError):
        value_at_risk(returns, 1.2)  # Invalid confidence level

    with pytest.raises(ValueError):
        value_at_risk(pd.Series([]), 0.95)  # Empty input


# Test conditional_value_at_risk
@pytest.mark.parametrize("confidence_level, expected", [(0.95, -0.0325), (0.99, -0.04)])
def test_conditional_value_at_risk(confidence_level, expected):
    result = conditional_value_at_risk(returns, confidence_level)
    assert np.isclose(result, expected, atol=1e-2)


def test_conditional_value_at_risk_invalid_input():
    with pytest.raises(ValueError):
        conditional_value_at_risk([1, 2, 3], 0.95)  # Invalid input type

    with pytest.raises(ValueError):
        conditional_value_at_risk(returns, 1.2)  # Invalid confidence level

    with pytest.raises(ValueError):
        conditional_value_at_risk(pd.Series([]), 0.95)  # Empty input


def test_maximum_drawdown_invalid_input():
    with pytest.raises(ValueError):
        maximum_drawdown([100, 105, 102], True)  # Invalid input type

    with pytest.raises(ValueError):
        maximum_drawdown(pd.Series([]), True)  # Empty input


# Edge cases
@pytest.mark.parametrize("input_data, expected", [
    (pd.Series([1, 1, 1]), 0.0),  # No drawdown
    (pd.Series([10, 5, 3]), -0.7)  # Continuous decline
])
def test_maximum_drawdown_edge_cases(input_data, expected):
    result = maximum_drawdown(input_data, False)
    assert np.isclose(result, expected, atol=1e-6)

if __name__ == "__main__":
    pytest.main()
