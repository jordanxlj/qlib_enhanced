# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import pandas as pd
import numpy as np
from qlib.contrib.strategy.historic_portfolio_volatility import (
    generate_weights,
    historical_covariance,
    get_annualization_factor,
    portfolio_volatility,
    compute_portfolio_metrics,
    calculate_returns,
    resample_returns,
)
from unittest.mock import patch

@pytest.fixture
def sample_returns():
    dates = pd.date_range('2023-01-01', periods=10)
    data = {
        'A': [np.nan, 0.01, 0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.04, -0.01],
        'B': [np.nan, 0.02, 0.01, 0.00, -0.01, 0.03, 0.02, 0.01, -0.02, 0.03],
    }
    returns = pd.DataFrame(data, index=dates)
    return returns

def test_generate_weights_ma():
    weights = generate_weights(5, method='ma')
    np.testing.assert_array_almost_equal(weights, np.full(5, 0.2))

def test_generate_weights_xma():
    weights = generate_weights(5, method='xma', half_life=2)
    decay = np.exp(np.log(0.5) / 2)
    expected = np.power(decay, np.arange(5))[::-1]
    expected /= expected.sum()
    np.testing.assert_array_almost_equal(weights, expected)

def test_generate_weights_invalid_method():
    with pytest.raises(ValueError):
        generate_weights(5, method='invalid')



def test_historical_covariance(sample_returns):
    cov = historical_covariance(sample_returns, window=5, method='ma')
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape[0] == 40  # 10 times, 2 inst, 2 inst
    # Numerical validation for a specific time point
    t = sample_returns.index[4]
    cov_matrix = cov.xs(t, level=0)['covariance'].unstack()
    expected_cov_AA = np.nanvar(sample_returns['A'].iloc[0:5], ddof=0)  # population variance since weighted_cov uses sum(w) as denom
    assert cov_matrix.loc['A', 'A'] == pytest.approx(expected_cov_AA, rel=1e-6)
    # Add more checks as needed

def test_historical_covariance_back_fill(sample_returns):
    cov = historical_covariance(sample_returns.iloc[:3], window=5, back_fill=True)
    assert cov.shape[0] == 3 * 2 * 2

def test_get_annualization_factor():
    assert get_annualization_factor('day') == pytest.approx(np.sqrt(252))
    assert get_annualization_factor('week') == pytest.approx(np.sqrt(52))
    assert get_annualization_factor('month') == pytest.approx(np.sqrt(12))
    with pytest.raises(ValueError):
        get_annualization_factor('invalid')

def test_portfolio_volatility():
    dates = pd.date_range('2023-01-01', periods=3)
    weights = pd.DataFrame({'A': [0.5, 0.5, 0.5], 'B': [0.5, 0.5, 0.5]}, index=dates)
    cov_data = []
    for t in dates:
        df = pd.DataFrame({
            'instrument1': ['A', 'A', 'B', 'B'],
            'instrument2': ['A', 'B', 'A', 'B'],
            'covariance': [1.0, 0.5, 0.5, 1.0],
        })
        df['datetime'] = t
        cov_data.append(df.set_index(['datetime', 'instrument1', 'instrument2']))
    cov_df = pd.concat(cov_data)
    cov_df = cov_df.sort_index()
    
    vol = portfolio_volatility(weights, cov_df)
    expected_vol = np.sqrt(0.5**2 * 1 + 0.5**2 * 1 + 2*0.5*0.5*0.5)  # = sqrt(0.75)
    np.testing.assert_array_almost_equal(vol, np.full(3, np.sqrt(0.75)))

@patch('qlib.contrib.strategy.historic_portfolio_volatility.D')
def test_compute_portfolio_metrics(mock_D, sample_returns):
    dates = sample_returns.index
    instruments = ['A', 'B']
    # Use fixed prices derived from returns
    prices_A = np.exp(np.cumsum(sample_returns['A'].fillna(0)))
    prices_B = np.exp(np.cumsum(sample_returns['B'].fillna(0)))
    
    # Create mock data with correct structure - instrument as index, datetime as columns
    mock_data = pd.DataFrame({
        '$close': np.concatenate([prices_A, prices_B])
    }, index=pd.MultiIndex.from_product([instruments, dates], names=['instrument', 'datetime']))
    mock_data = mock_data.sort_index()
    mock_D.features.return_value = mock_data

    instruments_list = instruments
    start_time = dates[0]
    end_time = dates[-1]
    metrics = compute_portfolio_metrics(instruments_list, start_time, end_time, frequency='day', vol_window=5, method='ma')
    assert 'volatility' in metrics
    assert 'covariance' in metrics
    assert metrics['volatility'].shape == (10, 2)
    
    # Numerical validation for volatility
    expected_cov_A = np.nanvar(sample_returns['A'].iloc[-5:], ddof=0)
    print(f"Calculated cov: {metrics['covariance'].loc[(dates[-1], 'A', 'A'), 'covariance']}")
    print(f"Expected cov: {expected_cov_A}")
    expected_vol_A = np.sqrt(expected_cov_A) * np.sqrt(252)
    assert metrics['volatility']['A'].iloc[-1] == pytest.approx(expected_vol_A, rel=1e-4)
    
    weights = pd.DataFrame(np.full((10, 2), 0.5), index=dates, columns=instruments)
    metrics_with_port = compute_portfolio_metrics(instruments_list, start_time, end_time, frequency='day', vol_window=5, weights=weights, method='ma')
    assert 'portfolio_vol' in metrics_with_port
    assert len(metrics_with_port['portfolio_vol']) == 10
    
    # Numerical validation for portfolio vol
    # For simplicity, check the last value
    last_cov = metrics_with_port['covariance'].xs(dates[-1], level=0)['covariance'].unstack()
    w = np.array([0.5, 0.5])
    expected_port_vol = np.sqrt(w.T @ last_cov.values @ w) * np.sqrt(252)
    assert metrics_with_port['portfolio_vol'].iloc[-1] == pytest.approx(expected_port_vol, rel=1e-4)

def test_resample_returns(sample_returns):
    weekly = resample_returns(sample_returns, 'week')
    assert len(weekly) <= 3  # Depending on dates
    with pytest.raises(ValueError):
        resample_returns(sample_returns, 'invalid')