"""
MLP training experiments on synthetic datasets (moons and linear_gaussian).
Trains MLP models and generates decision boundary plots.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MLP
from train import train
from optimizers import Adam
import utils
from utils import load_dataset
from plots import plot_decision_boundary, plot_training_dynamics

# Setup paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "results"
# FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_mlp_synthetic_experiment(
    name: str,
    dataset_name: str,
    hidden_dim: int = 32,
    lr: float = 0.05,
    epochs: int = 200,
    batch_size: int = 64,
    lam: float = 1e-4,
):
    """
    Train MLP on a synthetic dataset and create decision boundary plots.
    
    Args:
        name: Human-readable name for the dataset
        dataset_name: Name of dataset file (without .npz)
        hidden_dim: Number of hidden neurons
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size for training
        lam: L2 regularization coefficient
    """
    print(f"\n--- MLP experiment: {name} ---")
    
    # Load dataset (train, val, test)
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_dataset(dataset_name, train=True, val=True, test=True)
    
    d = X_tr.shape[1]  # Input dimension
    k = len(np.unique(y_tr))  # Number of classes
    print(f" d: {d}, k: {k}")
    print(f"  Train={len(y_tr)}, Val={len(y_v)}, Test={len(y_te)}")
    print(f"  Input dim={d}, Hidden dim={hidden_dim}, Classes={k}")
    
    # Initialize model and optimizer
    model = MLP(input_dim=d, hidden_dim=hidden_dim, num_classes=k)
    optimizer = Adam()

    # Train model
    best_params, history, best_epoch = train(
        model,
        optimizer,
        X_tr,
        y_tr,
        X_v,
        y_v,
        epochs=epochs,
        batch_size=batch_size,
        lam=lam
    )
    
    # Restore best parameters
    model.W1, model.b1 = best_params['W1'], best_params['b1']
    model.W2, model.b2 = best_params['W2'], best_params['b2']
    
    # Compute accuracies
    val_acc = utils.compute_accuracy(model, X_v, y_v)
    test_acc = utils.compute_accuracy(model, X_te, y_te)

    print(f"\n  Val accuracy:  {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    print(f"\nBest epoch: {best_epoch}")
    
    # Generate tag for filenames
    tag = name.lower().replace(" ", "_")
    
    # Plot decision boundaries
    print("\nGenerating plots...")
    
    plot_decision_boundary(
        model,
        X_tr,
        y_tr,
        title=f"MLP — {name} (train)",
        save_path=FIG_DIR / f"mlp_{tag}_train_boundary.png",
    )
    
    plot_decision_boundary(
        model,
        X_te,
        y_te,
        title=f"MLP — {name} (test)",
        save_path=FIG_DIR / f"mlp_{tag}_test_boundary.png",
    )


    return {
        'model': model,
        'history': history,
        'best_epoch': best_epoch,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'X_train': X_tr,
        'y_train': y_tr,
        'X_val': X_v,
        'y_val': y_v,
        'X_test': X_te,
        'y_test': y_te,
    }


def main():
    """Run all MLP experiments."""
    print("=" * 60)
    print("MLP EXPERIMENTS ON SYNTHETIC DATASETS")
    print("=" * 60)
    
    # Moons dataset
    moons_results = run_mlp_synthetic_experiment(
        name="Moons",
        dataset_name="moons",
        hidden_dim=32,
        lr=0.05,
        epochs=200,
        batch_size=64,
        lam=1e-4,
    )
    
    # Linear Gaussian dataset
    linear_gaussian_results = run_mlp_synthetic_experiment(
        name="Linear Gaussian",
        dataset_name="linear_gaussian",
        hidden_dim=32,
        lr=0.05,
        epochs=200,
        batch_size=64,
        lam=1e-4,
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  - mlp_moons_train_boundary.png")
    print("  - mlp_moons_test_boundary.png")
    print("  - mlp_moons_training_dynamics.png")
    print("  - mlp_linear_gaussian_train_boundary.png")
    print("  - mlp_linear_gaussian_test_boundary.png")
    print("  - mlp_linear_gaussian_training_dynamics.png")

    return {
        'moons': moons_results,
        'linear_gaussian': linear_gaussian_results,
    }


if __name__ == "__main__":
    results = main()

