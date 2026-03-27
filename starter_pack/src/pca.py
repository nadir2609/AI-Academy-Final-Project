import numpy as np


class PCA:
    """
    Principal Component Analysis via SVD.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self._all_s = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

        self.components_ = Vt[:self.n_components]
        total_var = (s ** 2).sum()
        self.explained_variance_ = s[:self.n_components] ** 2 / (len(X) - 1)
        self.explained_variance_ratio_ = s[:self.n_components] ** 2 / total_var
        self._all_s = s
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)