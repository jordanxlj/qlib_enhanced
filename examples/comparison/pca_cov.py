import numpy as np
from sklearn.decomposition import PCA

# 假设 X (t x p)
t, p = 100, 50
X = np.random.randn(t, p)  # 模拟数据

# PCA
pca = PCA(n_components=5)  # k=5
pca.fit(X)

F = pca.components_.T  # F (p x k), loadings
B = pca.transform(X)  # B = X F (t x k), scores
Phi = np.diag(pca.explained_variance_)  # Φ ≈ Λ_k
U = X - B @ F.T  # Residuals
Psi = np.diag(np.var(U, axis=0))  # Ψ diagonal

Sigma_hat = F @ Phi @ F.T + Psi  # Structured covariance