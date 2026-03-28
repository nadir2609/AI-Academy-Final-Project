"""
Capacity Ablation on Linear Gaussian Dataset

This script trains 3 MLPs with different hidden widths {2, 8, 32}
to analyze how model capacity affects generalization on the linear Gaussian dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from mlp import MLP
from train import train
from optimizers import Adam
import utils


def run_capacity_ablation():
    """Run capacity ablation experiment with hidden widths {2, 8, 32}."""
    np.random.seed(42)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = utils.load_dataset("linear_gaussian", test=True)

    # Configuration
    hidden_widths = [2, 8, 32]
    input_dim = 2  # Linear Gaussian has 2 features
    num_classes = 2  # Binary classification

    # Training hyperparameters
    epochs = 500
    batch_size = 32
    learning_rate = 0.001  # Adam typically uses smaller learning rate
    lam = 1e-4

    # Results output directory
    results_dir = os.path.join(SCRIPT_DIR, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Store results
    results = []
    trained_models = []

    print("=" * 70)
    print("Capacity Ablation on Linear Gaussian Dataset")
    print("=" * 70)
    print(f"{'Hidden Width':<15}{'Train Acc':<12}{'Val Acc':<12}{'Test Acc':<12}{'Boundary'}")
    print("-" * 70)

    boundary_labels = {2: 'Simple line', 8: 'Moderate curve', 32: 'Complex curve'}

    for hidden_dim in hidden_widths:
        # Initialize model and optimizer
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
        optimizer = Adam(lr=learning_rate)

        # Train model
        best_params, history, best_epoch = train(
            model, optimizer,
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            lam=lam
        )

        # Load best parameters
        model.W1 = best_params['W1']
        model.b1 = best_params['b1']
        model.W2 = best_params['W2']
        model.b2 = best_params['b2']

        # Compute accuracies
        train_acc = utils.compute_accuracy(model, X_train, y_train)
        val_acc = utils.compute_accuracy(model, X_val, y_val)
        test_acc = utils.compute_accuracy(model, X_test, y_test)

        results.append({
            'hidden_width': hidden_dim,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'history': history
        })
        trained_models.append(model)

        print(
            f"{hidden_dim:<15}{train_acc * 100:>10.1f}%{val_acc * 100:>10.1f}%{test_acc * 100:>10.1f}%    {boundary_labels[hidden_dim]}")

    print("=" * 70)

    # Generate decision boundary plots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.hstack([y_train, y_val, y_test])

    for idx, (model, res) in enumerate(zip(trained_models, results)):
        hidden_dim = res['hidden_width']
        test_acc = res['test_acc']

        utils.plot_decision_boundary(
            model, X_all, y_all, ax=axes[idx],
            title=f"Hidden Width = {hidden_dim}\nTest Acc: {test_acc * 100:.1f}%"
        )

    plt.suptitle("Capacity Ablation: Decision Boundaries on Linear Gaussian Dataset", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'capacity_ablation_linear_gaussian_boundaries.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved decision boundaries to: {output_path}")
    plt.show()

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for res in results:
        hidden_dim = res['hidden_width']
        history = res['history']

        axes[0].plot(history['train_loss'], label=f'Width={hidden_dim}')
        axes[1].plot(history['val_acc'], label=f'Width={hidden_dim}')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Validation Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Capacity Ablation: Training Dynamics on Linear Gaussian", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'capacity_ablation_linear_gaussian_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to: {output_path}")
    plt.show()

    # Analysis summary
    print("\n" + "=" * 70)
    print("Analysis: How does capacity affect generalization on Linear Gaussian?")
    print("=" * 70)
    print("\nKey Observations:")
    print("- Linear Gaussian has linearly separable classes")
    print("- Even small capacity (h=2) should achieve high accuracy")
    print("- Larger capacity may not provide significant benefit")
    print("- This demonstrates when simple models are sufficient")

    # Compare results
    print(f"\nComparison:")
    for res in results:
        print(f"  Width={res['hidden_width']:2d}: Train={res['train_acc'] * 100:.1f}%, "
              f"Val={res['val_acc'] * 100:.1f}%, Test={res['test_acc'] * 100:.1f}%")

    return results, trained_models


if __name__ == '__main__':
    results, models = run_capacity_ablation()
