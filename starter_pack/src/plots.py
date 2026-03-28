from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_decision_boundary(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        save_path: Path,
        transform=None,
):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    if transform is not None:
        grid = transform(grid)

    Z = model.predict(grid).reshape(xx.shape)

    cmap_bg = plt.cm.RdYlBu
    cmap_pts = plt.cm.Dark2

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)

    unique_classes = np.unique(y)
    for cls in unique_classes:
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            label=f"Class {cls}",
            s=25,
            edgecolors="k",
            linewidths=0.4,
            color=cmap_pts(cls / max(unique_classes.max(), 1)),
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_dynamics(history: dict, title: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].axvline(
        history.get("best_epoch", 0) + 1,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Best epoch",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], color="green", label="Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Val Accuracy")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
