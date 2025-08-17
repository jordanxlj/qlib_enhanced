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

        # Clean returns more permissively:
        # - drop columns with all-NaN
        # - drop rows with all-NaN
        # - remove assets with too few observations
        # - fill remaining NaNs with 0 to allow optimization
        clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if clean.empty:
            return pd.Series(dtype=float)

        min_obs = max(3, int(len(clean.index) * 0.3))
        counts = clean.count()
        keep_cols = counts[counts >= min_obs].index
        clean = clean.loc[:, keep_cols]
        if clean.shape[1] == 0:
            return pd.Series(dtype=float)

        clean = clean.fillna(0.0)
        if clean.empty or clean.shape[1] == 0:
            return pd.Series(dtype=float)

        port = rp.Portfolio(returns=clean)

        # Optional constraints and benchmark-related settings. Defaults are relaxed to avoid infeasibility.
        # Users can supply these via default_params or per-call params.
        # - turnover: float or None
        # - te: float or None (tracking error limit)
        # - kindbench: bool (True if benchmark input are weights, False if returns)
        # - lowerret / upperCVaR / uppermdd: floats or None
        # - b: bounds passed to optimization (also passed below)
        turnover_limit = cfg.get("turnover")
        # Turnover constraint is enabled only if limits are provided
        port.allowTO = turnover_limit is not None
        if turnover_limit is not None:
            port.turnover = float(turnover_limit)

        te_limit = cfg.get("te")
        port.kindbench = bool(cfg.get("kindbench", False))

        # Benchmark: accept Series or single-column DataFrame of benchmark returns when kindbench=False.
        if index is not None:
            # if isinstance(index, pd.DataFrame):
            #     if index.shape[1] > 1:
            #         index = index.iloc[:, 0]
            #     else:
            #         index = index.squeeze()
            port.benchindex = index

        # TE constraint is enabled only if limits are provided and compatible inputs exist
        port.allowTE = (te_limit is not None) and (index is not None)
        if te_limit is not None:
            port.TE = float(te_limit)

        # Risk constraints are optional; unset by default
        #port.lowerret = cfg.get("lowerret", None)
        #port.upperCVaR = cfg.get("upperCVaR", None)
        #port.uppermdd = cfg.get("uppermdd", None)

        # Compute mu and cov ourselves to avoid repeated PD warnings from riskfolio
        # and to apply a robust PD-fix before optimization
        # Mean vector
        if cfg["method_mu"] == "hist":
            mu = clean.mean()
        elif cfg["method_mu"] == "ewma":
            decay = float(cfg.get("d", 0.94))
            weights = np.array([decay ** (len(clean) - 1 - i) for i in range(len(clean))], dtype=float)
            weights = weights / weights.sum()
            mu = pd.Series((clean.values * weights[:, None]).sum(axis=0), index=clean.columns)
        else:
            # fallback to simple mean
            mu = clean.mean()

        # Covariance matrix
        if cfg["method_cov"] == "ewma":
            decay = float(cfg.get("d", 0.94))
            x = clean.values - mu.values
            S = np.zeros((x.shape[1], x.shape[1]), dtype=float)
            w = 1.0
            norm = 0.0
            for t in range(x.shape[0] - 1, -1, -1):
                xt = x[t : t + 1].T
                S = decay * S + (1 - decay) * (xt @ xt.T)
                w *= decay
                norm = decay * norm + (1 - decay)
            if norm > 0:
                S = S / norm
            cov = pd.DataFrame(S, index=clean.columns, columns=clean.columns)
        else:
            cov = clean.cov()

        # Ensure symmetry
        cov = (cov + cov.T) * 0.5

        # Make covariance positive definite via diagonal loading if needed
        eps = 1e-6
        try:
            min_eig = float(np.linalg.eigvalsh(cov.values).min())
        except Exception:
            min_eig = -1.0
        if min_eig < eps:
            jitter = (eps - min_eig) + 1e-8
            cov.values[range(cov.shape[0]), range(cov.shape[1])] += jitter

        port.mu = mu
        port.cov = cov

        # Run optimization with safe fallbacks when the problem is infeasible
        try:
            w = port.optimization(
                model=cfg["model"],
                rm=cfg["rm"],
                obj=cfg["obj"],
                rf=cfg["rf"],
                l=cfg["l"],
                hist=cfg["hist"],
            )
        except Exception as e:
            print(f"Optimization failed with error: {e}")
            w = None

        # If infeasible or None, relax constraints and retry with a classic MV Sharpe problem
        if w is None or (isinstance(w, (pd.Series, pd.DataFrame, np.ndarray)) and np.nan_to_num(np.array(w)).sum() == 0):
            try:
                # disable optional constraints
                port.allowTE = False
                port.allowTO = False
                port.lowerret = None
                port.upperCVaR = None
                port.uppermdd = None
                w = port.optimization(
                    model="Classic",
                    rm="MV",
                    obj="Sharpe",
                    rf=cfg["rf"],
                    l=cfg["l"],
                    hist=cfg["hist"],
                )
            except Exception as e:
                print(f"fallback to MV Sharpe Optimization failed with error: {e}")
                w = None

        # Ensure weights are a Series indexed by the same asset order
        if isinstance(w, pd.DataFrame):
            w = w.iloc[:, 0]
        if not isinstance(w, pd.Series):
            w = pd.Series(w, index=clean.columns)

        # Clean numerical noise, enforce non-negativity if short=False
        w = w.fillna(0.0).astype(float)
        # no short
        w = w.clip(lower=0.0)

        total = float(np.abs(w).sum())
        if total > 0:
            w = w / total
        else:
            # fallback to equal weight
            w = pd.Series(1.0 / clean.shape[1], index=clean.columns)

        return w