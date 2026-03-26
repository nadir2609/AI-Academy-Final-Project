import numpy as np
import matplotlib.pyplot as plt


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