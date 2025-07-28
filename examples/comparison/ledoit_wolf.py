import numpy as np
from sklearn.covariance import LedoitWolf, empirical_covariance
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Parameters: n samples, p features (high-dimensional case where p close to n for instability)
n_samples = 100  # Number of observations
n_features = 50  # Number of assets/variables (high dim to show shrinkage benefit)

# Generate true covariance matrix (positive definite)
true_cov = np.random.randn(n_features, n_features)
true_cov = true_cov.T @ true_cov + np.eye(n_features) * 10  # Ensure positive definite

# Generate multivariate normal data
X = np.random.multivariate_normal(np.zeros(n_features), true_cov, n_samples)

# Compute sample covariance
sample_cov = empirical_covariance(X)

# Apply Ledoit-Wolf shrinkage
lw = LedoitWolf()
lw.fit(X)
shrunk_cov = lw.covariance_
shrinkage_param = lw.shrinkage_

print(f"Estimated shrinkage parameter (delta): {shrinkage_param:.4f}")

# Compare condition numbers (to show stability improvement)
cond_true = np.linalg.cond(true_cov)
cond_sample = np.linalg.cond(sample_cov)
cond_shrunk = np.linalg.cond(shrunk_cov)

print(f"Condition number - True: {cond_true:.2f}")
print(f"Condition number - Sample: {cond_sample:.2f}")
print(f"Condition number - Shrunk: {cond_shrunk:.2f}")

# Visualize the matrices
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(true_cov, cmap='viridis')
axs[0].set_title('True Covariance Matrix')
axs[1].imshow(sample_cov, cmap='viridis')
axs[1].set_title('Sample Covariance Matrix')
axs[2].imshow(shrunk_cov, cmap='viridis')
axs[2].set_title('Ledoit-Wolf Shrunk Covariance')
plt.show()

# Optional: Manual implementation of Ledoit-Wolf for educational purposes
def manual_ledoit_wolf(X):
    n, p = X.shape
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    S = (X_centered.T @ X_centered) / n  # Sample covariance

    # Average variance
    var = np.diag(S).mean()

    # Shrinkage target: constant variance identity (simple version; full LW uses constant correlation)
    F = var * np.eye(p)

    # Compute shrinkage parameter (simplified; actual LW has more complex formula)
    m = np.trace(S) / p
    d2 = np.linalg.norm(S - F, 'fro')**2 / p**2
    b2 = min(d2, np.sum((X_centered**2).sum(axis=1)**2) / n**2 / p**2)
    delta = b2 / d2

    # Shrunk covariance
    shrunk = delta * F + (1 - delta) * S
    return shrunk, delta

manual_shrunk, manual_delta = manual_ledoit_wolf(X)
print(f"Manual shrinkage parameter: {manual_delta:.4f}")