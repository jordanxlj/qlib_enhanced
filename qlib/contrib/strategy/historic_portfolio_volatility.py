# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from qlib.data import D
from qlib.utils.resam import resam_ts_data

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from prices DataFrame.
    
    Parameters:
    - prices: pd.DataFrame with datetime index and instruments as columns.
    
    Returns:
    - pd.DataFrame of log returns.
    """
    return np.log(prices / prices.shift(1))

def resample_returns(returns: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample returns to the specified frequency.
    
    Parameters:
    - returns: pd.DataFrame of returns.
    - frequency: str, 'day', 'week', or 'month'.
    
    Returns:
    - Resampled returns DataFrame.
    """
    if frequency == 'day':
        return returns
    elif frequency == 'week':
        return returns.resample('W').sum()
    elif frequency == 'month':
        return returns.resample('M').sum()
    else:
        raise ValueError("Unsupported frequency. Use 'day', 'week', or 'month'.")

def historical_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling historical volatility (std dev of returns).
    
    Parameters:
    - returns: pd.DataFrame of returns.
    - window: int, rolling window size.
    
    Returns:
    - pd.DataFrame of volatility.
    """
    return returns.rolling(window).std()

def generate_weights(n: int, method: str = 'ma', half_life: int = None) -> np.ndarray:
    """
    Generate weights array based on method.
    
    Parameters:
    - n: int, number of periods.
    - method: str, 'ma' for flat, 'xma' for exponential.
    - half_life: int, for 'xma' method.
    
    Returns:
    - np.ndarray of weights.
    """
    if method == 'ma':
        return np.full(n, 1.0 / n)
    elif method == 'xma':
        if half_life is None:
            raise ValueError("half_life required for 'xma' method.")
        decay = np.exp(np.log(0.5) / half_life)
        weights = np.power(decay, np.arange(n))[::-1]
        weights /= weights.sum()
        return weights
    else:
        raise ValueError("Unsupported method. Use 'ma' or 'xma'.")

def weighted_covariance(x: pd.Series, y: pd.Series, weights: np.ndarray) -> float:
    """
    Calculate weighted covariance between x and y.
    
    Parameters:
    - x, y: pd.Series of same length.
    - weights: np.ndarray of weights.
    
    Returns:
    - float covariance.
    """
    if len(x) != len(y) or len(x) != len(weights):
        raise ValueError("Input lengths mismatch.")
    w_sum = weights.sum()
    mean_x = (weights * x).sum() / w_sum
    mean_y = (weights * y).sum() / w_sum
    cov = (weights * x * y).sum() / w_sum - mean_x * mean_y
    return cov

def historical_covariance(returns: pd.DataFrame, window: int = 20, method: str = 'ma', half_life: int = None, back_fill: bool = True) -> pd.DataFrame:
    """
    Calculate rolling weighted covariance matrix for each time step.
    
    Parameters:
    - returns: pd.DataFrame of returns (rows: time, columns: instruments).
    - window: int, rolling window size.
    - method: str, 'ma' or 'xma'.
    - half_life: int, for 'xma'.
    - back_fill: bool, if True, compute for windows smaller than 'window' using available data.
    
    Returns:
    - pd.DataFrame with MultiIndex (datetime, instrument1, instrument2) and 'covariance' column.
    """
    instruments = returns.columns
    cov_list = []
    for t in range(len(returns)):
        start = max(0, t - window + 1)
        ret_window = returns.iloc[start:t+1]
        n_win = len(ret_window)
        if not back_fill and n_win < window:
            cov_matrix = pd.DataFrame(np.nan, index=instruments, columns=instruments)
        else:
            weights = generate_weights(n_win, method, half_life)
            cov_matrix = pd.DataFrame(index=instruments, columns=instruments)
            for i in instruments:
                for j in instruments:
                    cov_matrix.loc[i, j] = weighted_covariance(ret_window[i], ret_window[j], weights)
        cov_list.append(cov_matrix)
    
    cov_df = pd.concat(cov_list, keys=returns.index)
    return cov_df.stack().to_frame('covariance')

def get_annualization_factor(frequency: str) -> float:
    """
    Get annualization factor for volatility.
    """
    if frequency == 'day':
        return np.sqrt(252)  # Approx trading days
    elif frequency == 'week':
        return np.sqrt(52)
    elif frequency == 'month':
        return np.sqrt(12)
    else:
        raise ValueError("Unsupported frequency.")

def portfolio_volatility(weights: pd.DataFrame, cov_df: pd.DataFrame) -> pd.Series:
    """
    Calculate portfolio volatility given weights and covariance DataFrame.
    
    Parameters:
    - weights: pd.DataFrame with datetime index and instruments as columns.
    - cov_df: pd.DataFrame with MultiIndex (datetime, inst1, inst2).
    
    Returns:
    - pd.Series of portfolio volatility over time.
    """
    vol = []
    for t in weights.index:
        w = weights.loc[t].fillna(0).values  # Handle missing weights as 0
        instruments = weights.columns
        try:
            # Try to get covariance data for time t
            cov_t = cov_df.xs(t, level='datetime')
            # Unstack to get matrix form
            cov_matrix = cov_t['covariance'].unstack().reindex(index=instruments, columns=instruments, fill_value=0)
        except KeyError:
            # If no data for this time, create zero matrix
            cov_matrix = pd.DataFrame(0.0, index=instruments, columns=instruments)
        
        vol_t = np.sqrt(w.T @ cov_matrix.values @ w)
        vol.append(vol_t)
    return pd.Series(vol, index=weights.index)

def compute_portfolio_metrics(
    instruments: list,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    frequency: str = 'day',
    vol_window: int = 20,
    weights: pd.DataFrame = None,
    method: str = 'ma',
    half_life: int = None,
    back_fill: bool = True
) -> dict:
    """
    Compute volatility and covariance for given instruments and frequency.
    
    Parameters:
    - instruments: list of stock codes.
    - start_time, end_time: pd.Timestamp for data range.
    - frequency: str, 'day', 'week', 'month'.
    - vol_window: int, rolling window for vol and cov.
    - weights: Optional pd.DataFrame for portfolio volatility.
    - method: str, 'ma' or 'xma' for weighting.
    - half_life: int, for 'xma'.
    - back_fill: bool, use back method for initial periods.
    
    Returns:
    - dict with 'volatility' (pd.DataFrame), 'covariance' (pd.DataFrame), and optionally 'portfolio_vol' (pd.Series).
    """
    prices = D.features(instruments, ['$close'], start_time=start_time, end_time=end_time, freq=frequency)
    prices = prices['$close'].unstack(level=0).astype(float)
    
    returns = calculate_returns(prices)
    resampled_returns = resample_returns(returns, frequency)
    
    cov = historical_covariance(resampled_returns, vol_window, method, half_life, back_fill)
    
    # Volatility from diagonal of cov
    vol = pd.DataFrame(index=cov.index.get_level_values(0).unique(), columns=prices.columns)
    for t in vol.index:
        for inst in vol.columns:
            vol.loc[t, inst] = np.sqrt(cov.loc[(t, inst, inst), 'covariance'])
    
    ann_factor = get_annualization_factor(frequency)
    vol = vol * ann_factor  # Annualize volatility
    
    metrics = {'volatility': vol, 'covariance': cov}
    
    if weights is not None:
        port_vol = portfolio_volatility(weights, cov)
        port_vol = port_vol * ann_factor  # Annualize
        metrics['portfolio_vol'] = port_vol
    
    return metrics