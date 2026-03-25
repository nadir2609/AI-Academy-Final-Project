import numpy as np


def softmax(x):
    """
    Compute softmax probabilities in a numerically stable way.

    Args:
        x: (n, k) or (k,) array of logits

    Returns:
        Softmax-normalized probabilities with the same shape as x.
    """
    x = np.asarray(x)

    if x.ndim == 1:
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x)

    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)