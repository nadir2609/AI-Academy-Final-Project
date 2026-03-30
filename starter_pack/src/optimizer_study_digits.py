"""
Optimizer Study on Digits Dataset

This script compares three optimizers (SGD, Momentum, Adam) on the Digits dataset
using an MLP with 32 hidden units. It trains each optimizer, records metrics,
and generates a comparative analysis with visualizations.(accuracy and loss curves)

Optimizers tested:
- SGD: lr=0.05
- Momentum: lr=0.05, beta=0.9
- Adam: lr=0.001
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import MLP
from train import train
from optimizers import SGD, Momentum, Adam
import utils


def train_with_optimizer(optimizer_name, optimizer, X_train, y_train, X_val, y_val,
                         input_dim=64, hidden_dim=32, num_classes=10,
                         epochs=200, batch_size=64, lam=1e-4):
    """Train a model with a specific optimizer and return results."""

    print(f"\n{'=' * 60}")
    print(f"Training with {optimizer_name}")
    print(f"{'=' * 60}")

    # create model
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    best_params, history, best_epoch = train(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        lam=lam
    )

    # load best parameters back into model
    model.W1 = best_params['W1']
    model.b1 = best_params['b1']
    model.W2 = best_params['W2']
    model.b2 = best_params['b2']

    # final loss and accuracy
    final_val_loss = min(history['val_loss'])
    final_val_acc = history['val_acc'][history['val_loss'].index(final_val_loss)]

    print(f"\n{optimizer_name} Results:")
    print(f"  Final Val Loss: {final_val_loss:.6f}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Val Accuracy at Best Epoch: {final_val_acc:.4f}")

    return {
        'model': model,
        'history': history,
        'best_params': best_params,
        'best_epoch': best_epoch,
        'final_val_loss': final_val_loss,
        'final_val_acc': final_val_acc
    }


def plot_comparison(results_dict, save_path):
    """Plot loss curves for all optimizers on the same figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = {'SGD': 'blue', 'Momentum': 'green', 'Adam': 'red'}

    # Plot training loss
    for opt_name, results in results_dict.items():
        history = results['history']
        ax1.plot(history['train_loss'], label=f'{opt_name}',
                 color=colors[opt_name], alpha=0.7, linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot validation loss
    for opt_name, results in results_dict.items():
        history = results['history']
        ax2.plot(history['val_loss'], label=f'{opt_name}',
                 color=colors[opt_name], alpha=0.7, linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Optimizer Study on Digits Dataset (MLP-32)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {save_path}")
    plt.show()


def print_results_table(results_dict, X_test, y_test):
    """Print a formatted table of optimizer comparison results."""
    print("\n" + "=" * 95)
    print("OPTIMIZER COMPARISON RESULTS")
    print("=" * 95)
    print(f"{'Optimizer':<15} {'Final Val Loss':<18} {'Val Acc':<12} {'Best Epoch':<15} {'Test Acc':<12}")
    print("-" * 95)

    for opt_name, results in results_dict.items():
        model = results['model']
        final_val_loss = results['final_val_loss']
        final_val_acc = results['final_val_acc']
        best_epoch = results['best_epoch']

        # Compute test accuracy
        test_acc = utils.compute_accuracy(model, X_test, y_test)

        print(f"{opt_name:<15} {final_val_loss:<18.6f} {final_val_acc:<12.4f} {best_epoch:<15} {test_acc:<12.4f}")

    print("=" * 95)


def analyze_convergence(results_dict):
    """Analyze convergence speed and stability."""
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    for opt_name, results in results_dict.items():
        history = results['history']
        val_losses = np.array(history['val_loss'])

        # Find epoch when loss drops below certain threshold
        best_loss = results['final_val_loss']
        threshold = best_loss * 1.1  # 10% above best
        epochs_to_converge = np.where(val_losses < threshold)[0]
        first_convergence = epochs_to_converge[0] if len(epochs_to_converge) > 0 else len(val_losses)

        # Calculate stability (variance in last 50 epochs)
        stability_window = val_losses[-50:]
        stability_std = np.std(stability_window)

        # Check for oscillations (count direction changes)
        diffs = np.diff(val_losses)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        print(f"\n{opt_name}:")
        print(f"  Convergence Speed:")
        print(f"    - Epochs to reach 110% of best loss: {first_convergence}")
        print(f"    - Best epoch: {results['best_epoch']}")
        print(f"  Stability:")
        print(f"    - Val loss std (last 50 epochs): {stability_std:.6f}")
        print(f"    - Number of oscillations: {sign_changes}")
        print(f"  Final Performance:")
        print(f"    - Best val loss: {best_loss:.6f}")
        print(f"    - Final val accuracy: {results['final_val_acc']:.4f}")

    print("=" * 80)


def get_training_config():
    """Get training configuration parameters."""
    return {
        'input_dim': 64,
        'hidden_dim': 32,
        'num_classes': 10,
        'epochs': 200,
        'batch_size': 64,
        'lam': 1e-4
    }


def get_optimizers():
    """Get dictionary of optimizers to compare."""
    return {
        'SGD': SGD(lr=0.05),
        'Momentum': Momentum(lr=0.05, beta=0.9),
        'Adam': Adam(lr=0.001)
    }


def print_dataset_info(X_train, y_train, X_val, X_test):
    """Print dataset information."""
    print(f"\nDataset info:")
    print(f"  Input dimension: {X_train.shape[1]}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")


def print_model_config(config):
    """Print model configuration."""
    print(f"\nModel configuration:")
    print(f"  Architecture: MLP-{config['hidden_dim']}")
    print(f"  Hidden units: {config['hidden_dim']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  L2 regularization: {config['lam']}")


def print_optimizer_info():
    """Print optimizer information."""
    print(f"\nOptimizers to compare:")
    print(f"  1. SGD (lr=0.05)")
    print(f"  2. Momentum (lr=0.05, beta=0.9)")
    print(f"  3. Adam (lr=0.001)")


def train_all_optimizers(optimizers, config, X_train, y_train, X_val, y_val):
    """Train models with all optimizers and return results."""
    results = {}
    for opt_name, optimizer in optimizers.items():
        results[opt_name] = train_with_optimizer(
            optimizer_name=opt_name,
            optimizer=optimizer,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **config
        )
    return results


def plot_loss_comparison(results_dict, save_path):
    """Plot loss curves for all optimizers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = {'SGD': 'blue', 'Momentum': 'green', 'Adam': 'red'}

    # Plot training loss
    for opt_name, results in results_dict.items():
        history = results['history']
        ax1.plot(history['train_loss'], label=f'{opt_name}',
                 color=colors[opt_name], alpha=0.7, linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot validation loss
    for opt_name, results in results_dict.items():
        history = results['history']
        ax2.plot(history['val_loss'], label=f'{opt_name}',
                 color=colors[opt_name], alpha=0.7, linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Optimizer Study on Digits Dataset (MLP-32) - Loss',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Loss comparison plot saved to: {save_path}")
    plt.show()


def plot_accuracy_comparison(results_dict, save_path):
    """Plot accuracy curves for train and validation sets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = {'SGD': 'blue', 'Momentum': 'green', 'Adam': 'red'}

    # Plot training accuracy
    for opt_name, results in results_dict.items():
        history = results['history']
        ax1.plot(history['train_acc'], label=f'{opt_name}',
                 color=colors[opt_name], alpha=0.7, linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy', fontsize=12)
    ax1.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot validation accuracy
    for opt_name, results in results_dict.items():
        history = results['history']
        ax2.plot(history['val_acc'], label=f'{opt_name}',
                 color=colors[opt_name], alpha=0.7, linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.suptitle('Optimizer Study on Digits Dataset (MLP-32) - Accuracy',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Accuracy comparison plot saved to: {save_path}")
    plt.show()


def generate_all_plots(results, figures_dir):
    """Generate all comparison plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate loss plots
    loss_plot_path = figures_dir / 'optimizer_comparison_digits_loss.png'
    plot_loss_comparison(results, loss_plot_path)

    # Generate accuracy plots
    acc_plot_path = figures_dir / 'optimizer_comparison_digits_accuracy.png'
    plot_accuracy_comparison(results, acc_plot_path)


def main():
    """Main function to run the optimizer study."""
    print("\n" + "#" * 80)
    print("OPTIMIZER STUDY ON DIGITS DATASET")
    print("#" * 80)

    # Load data
    print("\nLoading Digits dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = utils.load_dataset("digits_data", test=True)
    print_dataset_info(X_train, y_train, X_val, X_test)

    # Get configuration
    config = get_training_config()
    print_model_config(config)

    # Get optimizers
    optimizers = get_optimizers()
    print_optimizer_info()

    # Train with each optimizer
    results = train_all_optimizers(optimizers, config, X_train, y_train, X_val, y_val)

    # Generate all comparison plots
    figures_dir = Path(__file__).parent.parent / 'figures'
    generate_all_plots(results, figures_dir)

    # Print results table
    print_results_table(results, X_test, y_test)

    # Analyze convergence
    analyze_convergence(results)

    print("\n" + "#" * 80)
    print("OPTIMIZER STUDY COMPLETED!")
    print("#" * 80 + "\n")


if __name__ == '__main__':
    main()
