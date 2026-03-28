import numpy as np

from utils import softmax


class SoftmaxRegression:
    """
    Multiclass softmax regression.

    Model: s(x) = W x + b, P = softmax(S)
    Loss: cross-entropy + L2 regularisation on W
    """

    def __init__(self, d: int, k: int, lam: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d + k))
        self.W = rng.standard_normal((k, d)) * scale
        self.b = np.zeros(k)
        self.lam = lam
        self.k = k

    def forward(self, X: np.ndarray):
        """
        Compute logits and probabilities for a batch X of shape (n, d).
        """
        S = X @ self.W.T + self.b
        P = softmax(S)
        return S, P

    def backward(self, X: np.ndarray, P: np.ndarray, Y_onehot: np.ndarray):
        """
        Compute gradients of W and b.
        """
        n = X.shape[0]
        dS = (P - Y_onehot) / n
        dW = dS.T @ X + self.lam * self.W
        db = dS.sum(axis=0)
        return dW, db

    def step(self, dW: np.ndarray, db: np.ndarray, lr: float):
        self.W -= lr * dW
        self.b -= lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, P = self.forward(X)
        return np.argmax(P, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, P = self.forward(X)
        return P
