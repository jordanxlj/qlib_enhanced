# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import pandas as pd
import numpy as np
from qlib.contrib.strategy.historic_portfolio_volatility import (
    generate_weights,
    weighted_covariance,
    historical_covariance,
    get_annualization_factor,
    portfolio_volatility,
    compute_portfolio_metrics,
    calculate_returns,
    resample_returns,
)

@pytest.fixture
def sample_returns():
    dates = pd.date_range('2023-01-01', periods=10)
    data = {
        'A': np.random.randn(10).cumsum(),
        'B': np.random.randn(10).cumsum(),
    }
    prices = pd.DataFrame(data, index=dates)
    returns = calculate_returns(prices)
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

def test_weighted_covariance():
    x = pd.Series([1, 2, 3])
    y = pd.Series([4, 5, 6])
    weights = np.array([0.2, 0.3, 0.5])
    cov = weighted_covariance(x, y, weights)
    assert cov == pytest.approx(1.0)  # Manual calc: weighted cov = 1.0

def test_weighted_covariance_mismatch_length():
    with pytest.raises(ValueError):
        weighted_covariance(pd.Series([1,2]), pd.Series([3,4,5]), np.array([0.1, 0.2]))

def test_historical_covariance(sample_returns):
    cov = historical_covariance(sample_returns, window=5, method='ma')
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape[0] == 10 * 2 * 2  # 10 times, 2 inst, 2 inst
    # Check one value
    t = sample_returns.index[4]
    cov_matrix = cov.loc[(t, slice(None), slice(None)), 'covariance'].unstack()
    assert cov_matrix.shape == (2, 2)

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
        cov_matrix = pd.DataFrame({'A': [1, 0.5], 'B': [0.5, 1]}, index=['A', 'B'])
        cov_data.append(cov_matrix.stack().to_frame('covariance').set_index(pd.Index([t]*4, name='datetime'), append=True).swaplevel(0,1))
    cov_df = pd.concat(cov_data)
    cov_df.index = pd.MultiIndex.from_tuples(cov_df.index)

    vol = portfolio_volatility(weights, cov_df)
    expected_vol = np.sqrt(0.5**2 * 1 + 0.5**2 * 1 + 2*0.5*0.5*0.5)  # = sqrt(0.75)
    np.testing.assert_array_almost_equal(vol, np.full(3, np.sqrt(0.75)))

def test_compute_portfolio_metrics(sample_returns):
    instruments = ['A', 'B']
    start_time = sample_returns.index[0]
    end_time = sample_returns.index[-1]
    metrics = compute_portfolio_metrics(instruments, start_time, end_time, frequency='day', vol_window=5, method='ma')
    assert 'volatility' in metrics
    assert 'covariance' in metrics
    assert metrics['volatility'].shape == (10, 2)

    weights = pd.DataFrame(np.full((10, 2), 0.5), index=sample_returns.index, columns=instruments)
    metrics_with_port = compute_portfolio_metrics(instruments, start_time, end_time, frequency='day', vol_window=5, weights=weights)
    assert 'portfolio_vol' in metrics_with_port
    assert len(metrics_with_port['portfolio_vol']) == 10

def test_resample_returns(sample_returns):
    weekly = resample_returns(sample_returns, 'week')
    assert len(weekly) <= 3  # Depending on dates
    with pytest.raises(ValueError):
        resample_returns(sample_returns, 'invalid')