import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Define the 7 countries
countries = ['Australia', 'Canada', 'France', 'Germany', 'Japan', 'UK', 'USA']

# Covariance matrix from the document (annual, assumed)
Sigma = np.array([
    [0.02560, 0.01585, 0.01897, 0.02233, 0.01475, 0.01638, 0.01469],
    [0.01585, 0.04121, 0.03343, 0.03603, 0.01322, 0.02469, 0.02957],
    [0.01897, 0.03343, 0.06150, 0.05787, 0.01849, 0.03884, 0.03098],
    [0.02233, 0.03603, 0.05787, 0.07344, 0.02015, 0.04211, 0.03309],
    [0.01475, 0.01322, 0.01849, 0.02015, 0.04410, 0.01701, 0.01202],
    [0.01638, 0.02469, 0.03884, 0.04211, 0.01701, 0.04000, 0.02439],
    [0.01469, 0.02957, 0.03098, 0.03309, 0.01202, 0.02439, 0.03497]
])

# Risk aversion coefficient
delta = 2.5

# Function to compute optimal portfolio weights (max utility: w.T mu - (delta/2) w.T Sigma w, sum w=1, w>=0)
def optimal_portfolio(mu, Sigma, delta):
    n = len(mu)
    def negative_utility(w):
        return - (w @ mu - (delta / 2) * (w @ Sigma @ w))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n)]
    result = minimize(negative_utility, np.ones(n) / n, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 1. Mean-Variance with uniform expected returns (5% for all)
mu_uniform = 0.05 * np.ones(7)
w_mv_uniform = optimal_portfolio(mu_uniform, Sigma, delta)
print("MV Uniform Weights:")
print(pd.Series(w_mv_uniform, index=countries))

# Expected return, risk, Sharpe (assuming rf=0 for simplicity)
exp_ret_mv_uniform = w_mv_uniform @ mu_uniform
risk_mv_uniform = np.sqrt(w_mv_uniform @ Sigma @ w_mv_uniform)
sharpe_mv_uniform = exp_ret_mv_uniform / risk_mv_uniform

# 2. Naive Mean-Variance with direct views incorporated (France, Germany, UK at 7%, others 5%)
mu_naive_view = 0.05 * np.ones(7)
mu_naive_view[[2, 3, 5]] = 0.07  # Indices: 2=France, 3=Germany, 5=UK
w_mv_view = optimal_portfolio(mu_naive_view, Sigma, delta)
print("\nMV with Naive Views Weights:")
print(pd.Series(w_mv_view, index=countries))

exp_ret_mv_view = w_mv_view @ mu_naive_view
risk_mv_view = np.sqrt(w_mv_view @ Sigma @ w_mv_view)
sharpe_mv_view = exp_ret_mv_view / risk_mv_view

# 3. Black-Litterman with Bayesian update
# Prior: mu ~ N(0.05 ones, tau * Sigma)
tau = 0.05  # Uncertainty in prior
pi = mu_uniform  # Prior mean (equilibrium returns)
prior_cov = tau * Sigma

# Views: Absolute views on France, Germany, UK at 7%
# P matrix (3 views x 7 assets)
P = np.zeros((3, 7))
P[0, 2] = 1  # France
P[1, 3] = 1  # Germany
P[2, 5] = 1  # UK
Q = 0.07 * np.ones(3)  # View means
view_uncertainty = 0.02  # Standard deviation of views (2%)
Omega = np.diag([view_uncertainty**2] * 3)  # View covariance

# Posterior calculations
inv_prior_cov = np.linalg.inv(prior_cov)
inv_Omega = np.linalg.inv(Omega)
post_precision = inv_prior_cov + P.T @ inv_Omega @ P
post_cov = np.linalg.inv(post_precision)
post_mu = post_cov @ (inv_prior_cov @ pi + P.T @ inv_Omega @ Q)

# Optimal portfolio with BL posterior mu
w_bl = optimal_portfolio(post_mu, Sigma, delta)
print("\nBlack-Litterman Weights:")
print(pd.Series(w_bl, index=countries))

exp_ret_bl = w_bl @ post_mu
risk_bl = np.sqrt(w_bl @ Sigma @ w_bl)
sharpe_bl = exp_ret_bl / risk_bl

# Comparison table
data = {
    'Metric': ['Weights', 'Expected Return', 'Risk (Volatility)', 'Sharpe Ratio (rf=0)'],
    'MV Uniform': [w_mv_uniform.round(4), round(exp_ret_mv_uniform, 4), round(risk_mv_uniform, 4), round(sharpe_mv_uniform, 4)],
    'MV Naive Views': [w_mv_view.round(4), round(exp_ret_mv_view, 4), round(risk_mv_view, 4), round(sharpe_mv_view, 4)],
    'Black-Litterman': [w_bl.round(4), round(exp_ret_bl, 4), round(risk_bl, 4), round(sharpe_bl, 4)]
}
comparison_df = pd.DataFrame(data)
print("\nComparison:")
print(comparison_df)