import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """Fit the model with X by computing the mean and principal components.
        """

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        n = X_centered.shape[0]
        features = X_centered.shape[1]

        assert features > self.n_components, "n_components must be <= number of features"

        covariance_matrix = (X_centered.T @ X_centered) / (n - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        eigenvectors = eigenvectors.T

        # sort eigenvalues and return its indices
        idxs = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # top n eigenvalues
        self.explained_variance_ = eigenvalues[:self.n_components]

        # sum of all eigenvalues
        total_variance = np.sum(eigenvalues)

        # explained variance ratio
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
