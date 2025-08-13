# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd
import riskfolio as rp  # type: ignore

class BasePositionManager(ABC):
    """Abstract base class for position managers.

    This class defines the interface for optimizing portfolio weights given
    a returns matrix. Subclasses must implement the optimize method.

    Notes
    -----
    - Weights returned should sum to 1 and be non-negative unless shorting is enabled.
    """

    def __init__(self, default_params: Optional[Dict] = None) -> None:
        self.default_params: Dict = default_params or {}

    @abstractmethod
    def optimize(self, returns: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
        """Optimize portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns with datetime index and asset tickers as columns.
        params : Dict, optional
            Custom parameters for the optimization method.

        Returns
        -------
        pd.Series
            Optimized weights indexed by asset ticker. Sum to 1.0.
        """
        pass


class RiskfolioPositionManager(BasePositionManager):
    """Position manager powered by riskfolio-lib.

    This class wraps common usage of riskfolio-lib's Portfolio API to produce
    optimized portfolio weights given a returns matrix.

    Notes
    -----
    - This class imports riskfolio-lib lazily. If the package is not installed,
      an informative ImportError will be raised when optimize is called.
    - Weights returned always sum to 1 and are non-negative unless `short` is
      enabled in parameters passed to `optimize`.
    """
    def __init__(self, default_params: Optional[Dict] = None, **kwargs) -> None:
        """Initialize the Riskfolio-based position manager.

        Parameters
        ----------
        default_params : Dict, optional
            Default parameters passed to the optimizer unless overridden in
            `optimize`. Examples include:
              - method_mu: 'hist' | 'ewma' | ...
              - method_cov: 'hist' | 'ewma' | ...
              - model: 'Classic' | 'BL' | 'FM' | ...
              - rm: 'MV' | 'MAD' | 'MSV' | ...
              - obj: 'Sharpe' | 'MinRisk' | ...
              - rf, l, short, hist, b
        **kwargs
            Shorthand for `default_params`. Any provided keyword arguments
            are merged into `default_params` and used as the default
            optimization parameters.
        """
        merged_params: Dict = dict(default_params or {})
        merged_params.update(kwargs)
        super().__init__(default_params=merged_params)

    def optimize(self, returns: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
        """Optimize weights using riskfolio-lib.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns with datetime index and asset tickers as columns.
        params : Dict, optional
            Parameters compatible with riskfolio Portfolio API:
              - method_mu: str, default 'hist'
              - method_cov: str, default 'hist'
              - d: float, default 0.94 (decay for EWMA when relevant)
              - model: str, default 'Classic'
              - rm: str, default 'MV' (risk measure)
              - obj: str, default 'Sharpe'
              - rf: float, default 0.0 (risk-free rate)
              - l: float, default 0.0 (risk aversion)
              - short: bool, default False
              - hist: bool, default True
              - b: tuple or None, bounds per-asset, optional

        Returns
        -------
        pd.Series
            Optimized weights indexed by asset ticker. Sum to 1.0.
        """
        if returns is None or returns.empty:
            return pd.Series(dtype=float)

        cfg: Dict = {**{
            "method_mu": "hist",
            "method_cov": "hist",
            "model": "Classic",
            "rm": "MV",
            "obj": "Sharpe",
            "rf": 0.0,
            "l": 0.0,
            "hist": True,
        }, **self.default_params, **(params or {})}

        # Clean returns: drop columns with all-NaN and rows with any NaN
        clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
        if clean.empty or clean.shape[1] == 0:
            return pd.Series(dtype=float)

        port = rp.Portfolio(returns=clean)
        port.assets_stats(
            method_mu=cfg["method_mu"],
            method_cov=cfg["method_cov"],
        )

        w = port.optimization(
            model=cfg["model"],
            rm=cfg["rm"],
            obj=cfg["obj"],
            rf=cfg["rf"],
            l=cfg["l"],
            hist=cfg["hist"],
        )

        # Ensure weights are a Series indexed by the same asset order
        if isinstance(w, pd.DataFrame):
            w = w.iloc[:, 0]
        if not isinstance(w, pd.Series):
            w = pd.Series(w, index=clean.columns)

        # Clean numerical noise, enforce non-negativity if short=False
        w = w.fillna(0.0).astype(float)
        if not cfg["short"]:
            w = w.clip(lower=0.0)
        total = float(np.abs(w).sum())
        if total > 0:
            w = w / total
        else:
            # fallback to equal weight
            w = pd.Series(1.0 / clean.shape[1], index=clean.columns)

        return w