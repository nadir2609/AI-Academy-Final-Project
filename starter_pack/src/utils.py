import numpy as np


def softmax(S: np.ndarray) -> np.ndarray:
    """
    Numerically stable row-wise softmax.
    """
    shift = S - S.max(axis=1, keepdims=True)
    E = np.exp(shift)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy_loss(
    P: np.ndarray,
    Y_onehot: np.ndarray,
    W: np.ndarray,
    lam: float,
) -> float:
    """
    Mean cross-entropy loss with L2 regularisation.
    """
    n = P.shape[0]
    log_P = np.log(np.clip(P, 1e-15, 1.0))
    ce = -np.sum(Y_onehot * log_P) / n
    reg = (lam / 2.0) * np.sum(W ** 2)
    return ce + reg


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    """
    Convert integer label vector to one-hot matrix of shape (n, k).
    """
    n = len(y)
    Y = np.zeros((n, k), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    return Y