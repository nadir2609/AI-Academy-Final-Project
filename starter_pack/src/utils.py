import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_dataset(dataset_name, train=True, val=True, test=False):
    """Load dataset from .npz file with flexible return options.

    Args:
        dataset_name: Name of the dataset file (without .npz extension)
        train: Whether to return training data (default: True)
        val: Whether to return validation data (default: True)
        test: Whether to return test data (default: False)

    Returns:
        Tuple of requested datasets in order: X_train, y_train, X_val, y_val, X_test, y_test
        Only returns the splits that are requested (set to True)
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    data_path = data_dir / f'{dataset_name}.npz'
    data = np.load(data_path)

    result = []

    # Check if data is already split
    if 'X_train' in data.keys():
        if train:
            result.extend([data['X_train'], data['y_train']])
        if val:
            result.extend([data['X_val'], data['y_val']])
        if test and 'X_test' in data.keys():
            result.extend([data['X_test'], data['y_test']])
    else:
        # Handle digits_data which needs to be split using indices
        X, y = data['X'], data['y']
        # Extract base name (e.g., 'digits' from 'digits_data')
        base_name = dataset_name.replace('_data', '')
        indices_path = data_dir / f'{base_name}_split_indices.npz'
        indices = np.load(indices_path)

        if train:
            train_idx = indices['train_idx']
            result.extend([X[train_idx], y[train_idx]])
        if val:
            val_idx = indices['val_idx']
            result.extend([X[val_idx], y[val_idx]])
        if test and 'test_idx' in indices.keys():
            test_idx = indices['test_idx']
            result.extend([X[test_idx], y[test_idx]])

    return tuple(result) if len(result) > 1 else result[0] if result else None


def compute_accuracy(model, X, y):
    """Evaluate model accuracy on a dataset."""
    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy


def plot_decision_boundary(model, X, y, ax=None, title=None, resolution=200):
    """
    Plot the decision boundary for a 2D classification model.

    Args:
        model: Trained model with a forward() method returning probabilities
        X: (n, 2) input data points
        y: (n,) true class labels
        ax: matplotlib axis to plot on (creates new figure if None)
        title: title for the plot
        resolution: grid resolution for the decision boundary

    Returns:
        ax: matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Define grid bounds with margin
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create meshgrid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict on grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = model.forward(grid_points)
    preds = np.argmax(probs, axis=1)
    preds = preds.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, preds, alpha=0.3, cmap='coolwarm')
    ax.contour(xx, yy, preds, colors='k', linewidths=0.5)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=30)

    if title:
        ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    return ax


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
