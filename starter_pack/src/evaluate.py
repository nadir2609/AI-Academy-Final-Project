import numpy as np

from .utils import one_hot


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Return accuracy and mean cross-entropy on a split.
    """
    k = model.k
    P = model.predict_proba(X)
    Y = one_hot(y, k)
    acc = np.mean(np.argmax(P, axis=1) == y)
    ce = -np.sum(Y * np.log(np.clip(P, 1e-15, 1.0))) / len(y)
    return {"accuracy": acc, "cross_entropy": ce}