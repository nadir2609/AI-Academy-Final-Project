"""
1. Use training set to fit parameters
2. Use validation set to choose best epoch (checkpoint)
3. Fix ONE final configuration for each model (already tuned)
4. Run 5 seeds for each configuration
5. Evaluate test set ONCE per seed (after training completes)
6. Report: mean ± 2.776 × (std / √5) for 95% CI with 4 degrees of freedom

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy

# Import our implementations
from model import MLP
from train import train
from optimizers import Adam, SGD
from utils import load_dataset, softmax


# =============================================================================
# SOFTMAX REGRESSION CLASS (simplified version for this script)
# =============================================================================

class SoftmaxRegression:
    """
    Multiclass softmax regression baseline.
    
    This is simpler than the neural network - just a linear model:
        scores = X @ W + b
        probabilities = softmax(scores)
    """

    def __init__(self, input_dim, num_classes, seed=0):
        np.random.seed(seed)
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + num_classes))
        self.W = np.random.randn(input_dim, num_classes) * scale
        self.b = np.zeros(num_classes)
        self.cache = {}

    def forward(self, X):
        """Forward pass: X -> scores -> probabilities"""
        scores = X @ self.W + self.b
        probs = softmax(scores)
        self.cache = {'X': X, 'probs': probs}
        return probs

    def backward(self, y_true):
        """Backward pass: compute gradients"""
        n = y_true.shape[0]
        X, probs = self.cache['X'], self.cache['probs']

        # One-hot encode labels
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n), y_true] = 1

        # Gradient of cross-entropy + softmax combined
        dscores = (probs - y_onehot) / n

        # Gradients for W and b
        dW = X.T @ dscores
        db = np.sum(dscores, axis=0)

        return {'W': dW, 'b': db}


# =============================================================================
# TRAINING FUNCTION FOR SOFTMAX
# =============================================================================

def train_softmax(model, X_train, y_train, X_val, y_val,
                  epochs=200, batch_size=64, lr=0.05, lam=1e-4, seed=0):
    """
    Train softmax regression with mini-batch SGD.
    Returns best model (by validation loss) and history.
    """
    np.random.seed(seed)
    n = X_train.shape[0]
    eps = 1e-8

    best_val_loss = float('inf')
    best_W = model.W.copy()
    best_b = model.b.copy()
    best_epoch = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def compute_loss_acc(X, y):
        probs = model.forward(X)
        # Cross-entropy loss
        loss = -np.mean(np.log(probs[np.arange(len(y)), y] + eps))
        # Add L2 regularization
        loss += 0.5 * lam * np.sum(model.W ** 2)
        # Accuracy
        acc = np.mean(np.argmax(probs, axis=1) == y)
        return loss, acc

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        X_shuf, y_shuf = X_train[indices], y_train[indices]

        # Mini-batch SGD
        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch, y_batch = X_shuf[start:end], y_shuf[start:end]

            model.forward(X_batch)
            grads = model.backward(y_batch)

            # Add L2 regularization gradient
            grads['W'] += lam * model.W

            # SGD update
            model.W -= lr * grads['W']
            model.b -= lr * grads['b']

        # Compute epoch metrics
        train_loss, train_acc = compute_loss_acc(X_train, y_train)
        val_loss, val_acc = compute_loss_acc(X_val, y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Save best checkpoint (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_W = model.W.copy()
            best_b = model.b.copy()
            best_epoch = epoch

    # Restore best weights
    model.W = best_W
    model.b = best_b

    return model, history, best_epoch


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model, X, y, model_type='mlp'):
    """
    Evaluate a trained model on given data.
    
    Returns:
        accuracy: fraction of correct predictions
        cross_entropy: mean negative log-likelihood of true class
    """
    eps = 1e-8

    # Get predictions
    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)

    # Compute metrics
    accuracy = np.mean(predictions == y)

    # Cross-entropy: -log(probability of true class)
    true_class_probs = probs[np.arange(len(y)), y]
    cross_entropy = -np.mean(np.log(true_class_probs + eps))

    return accuracy, cross_entropy


# =============================================================================
# CONFIDENCE INTERVAL CALCULATION
# =============================================================================

def compute_confidence_interval(values, confidence=0.95):
    """
    Compute 95% confidence interval for the mean.
    
    Formula: mean ± t_critical × (std / √n)
    
    For n=5 samples and 95% confidence:
        degrees of freedom = n - 1 = 4
        t_critical (two-tailed, 95%) = 2.776
    
    This tells us: "We are 95% confident the true mean lies in this range"
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # ddof=1 for sample std (Bessel's correction)

    # t-critical value for 95% CI with n-1 degrees of freedom
    # For n=5: t_critical = 2.776
    t_critical = 2.776

    # Standard error of the mean
    sem = std / np.sqrt(n)

    # Confidence interval
    margin = t_critical * sem
    ci_lower = mean - margin
    ci_upper = mean + margin

    return {
        'mean': mean,
        'std': std,
        'sem': sem,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'margin': margin
    }


# =============================================================================
# MAIN REPEATED-SEED EVALUATION
# =============================================================================

def run_repeated_seed_evaluation(seeds=[0, 1, 2, 3, 4]):
    """
    Run the complete repeated-seed evaluation protocol.
    
    For both Softmax Regression and MLP:
      1. Train with each seed
      2. Select best epoch by validation loss
      3. Evaluate on test set
      4. Compute statistics across all seeds
    """

    print("=" * 70)
    print("REPEATED-SEED EVALUATION ON DIGITS BENCHMARK")
    print("=" * 70)
    print(f"\nSeeds used: {seeds}")
    print(f"Number of runs: {len(seeds)}")

    # Load data
    print("\nLoading digits dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("digits_data", test=True)

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # Configuration (from assignment defaults)
    config = {
        'input_dim': 64,
        'hidden_dim': 32,
        'num_classes': 10,
        'epochs': 200,
        'batch_size': 64,
        'lam': 1e-4,
        'softmax_lr': 0.05,  # SGD for softmax
        'mlp_lr': 0.001,  # Adam for MLP
    }

    print(f"\nConfiguration:")
    print(f"  Hidden units: {config['hidden_dim']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  L2 regularization: {config['lam']}")

    # Storage for results
    results = {
        'softmax': {'accuracy': [], 'cross_entropy': [], 'best_epochs': []},
        'mlp': {'accuracy': [], 'cross_entropy': [], 'best_epochs': []}
    }

    # =========================================================================
    # RUN SOFTMAX REGRESSION WITH EACH SEED
    # =========================================================================
    print("\n" + "-" * 70)
    print("SOFTMAX REGRESSION (5 seeds)")
    print("-" * 70)

    for seed in seeds:
        print(f"\n  Seed {seed}:", end=" ")

        # Create and train model
        model = SoftmaxRegression(
            input_dim=config['input_dim'],
            num_classes=config['num_classes'],
            seed=seed
        )

        model, history, best_epoch = train_softmax(
            model, X_train, y_train, X_val, y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['softmax_lr'],
            lam=config['lam'],
            seed=seed
        )

        # Evaluate on TEST set (only after training is complete!)
        test_acc, test_ce = evaluate_model(model, X_test, y_test, 'softmax')

        results['softmax']['accuracy'].append(test_acc)
        results['softmax']['cross_entropy'].append(test_ce)
        results['softmax']['best_epochs'].append(best_epoch)

        print(f"Best epoch={best_epoch:3d}, Test Acc={test_acc:.4f}, Test CE={test_ce:.4f}")

    # =========================================================================
    # RUN MLP WITH EACH SEED
    # =========================================================================
    print("\n" + "-" * 70)
    print("MLP (Neural Network) with 32 hidden units (5 seeds)")
    print("-" * 70)

    for seed in seeds:
        print(f"\n  Seed {seed}:", end=" ")

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Create model and optimizer
        model = MLP(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes']
        )

        optimizer = Adam(lr=config['mlp_lr'])

        # Train model
        best_params, history, best_epoch = train(
            model, optimizer,
            X_train, y_train, X_val, y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lam=config['lam']
        )

        # Load best parameters
        model.W1 = best_params['W1']
        model.b1 = best_params['b1']
        model.W2 = best_params['W2']
        model.b2 = best_params['b2']

        # Evaluate on TEST set
        test_acc, test_ce = evaluate_model(model, X_test, y_test, 'mlp')

        results['mlp']['accuracy'].append(test_acc)
        results['mlp']['cross_entropy'].append(test_ce)
        results['mlp']['best_epochs'].append(best_epoch)

        print(f"Best epoch={best_epoch:3d}, Test Acc={test_acc:.4f}, Test CE={test_ce:.4f}")

    # =========================================================================
    # COMPUTE STATISTICS AND CONFIDENCE INTERVALS
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 70)

    stats = {}

    for model_name in ['softmax', 'mlp']:
        print(f"\n{model_name.upper()}:")

        # Compute CI for accuracy
        acc_stats = compute_confidence_interval(results[model_name]['accuracy'])
        ce_stats = compute_confidence_interval(results[model_name]['cross_entropy'])

        stats[model_name] = {'accuracy': acc_stats, 'cross_entropy': ce_stats}

        print(f"  Test Accuracy:     {acc_stats['mean'] * 100:.2f}% ± {acc_stats['margin'] * 100:.2f}%")
        print(
            f"                     (95% CI: [{acc_stats['ci_lower'] * 100:.2f}%, {acc_stats['ci_upper'] * 100:.2f}%])")
        print(f"  Test Cross-Entropy: {ce_stats['mean']:.4f} ± {ce_stats['margin']:.4f}")
        print(f"                     (95% CI: [{ce_stats['ci_lower']:.4f}, {ce_stats['ci_upper']:.4f}])")
        print(f"  Std Dev (Accuracy): {acc_stats['std'] * 100:.2f}%")
        print(f"  Individual runs:   {[f'{x * 100:.1f}%' for x in results[model_name]['accuracy']]}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    softmax_acc = stats['softmax']['accuracy']['mean']
    mlp_acc = stats['mlp']['accuracy']['mean']
    diff = mlp_acc - softmax_acc

    print(f"\n  MLP vs Softmax accuracy difference: {diff * 100:+.2f}%")

    # Check if confidence intervals overlap
    s_lower = stats['softmax']['accuracy']['ci_lower']
    s_upper = stats['softmax']['accuracy']['ci_upper']
    m_lower = stats['mlp']['accuracy']['ci_lower']
    m_upper = stats['mlp']['accuracy']['ci_upper']

    if m_lower > s_upper:
        print("  → MLP is SIGNIFICANTLY better (CIs don't overlap)")
    elif s_lower > m_upper:
        print("  → Softmax is SIGNIFICANTLY better (CIs don't overlap)")
    else:
        print("  → Difference is NOT statistically significant (CIs overlap)")

    # =========================================================================
    # GENERATE SUMMARY TABLE FOR REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("TABLE FOR REPORT (Copy this)")
    print("=" * 70)
    print("""
+------------------+------------------------+------------------------+
| Model            | Test Accuracy          | Test Cross-Entropy     |
+------------------+------------------------+------------------------+""")

    for model_name in ['softmax', 'mlp']:
        acc = stats[model_name]['accuracy']
        ce = stats[model_name]['cross_entropy']
        name = "Softmax Regression" if model_name == 'softmax' else "MLP (h=32)"
        print(
            f"| {name:<16} | {acc['mean'] * 100:.2f}% ± {acc['margin'] * 100:.2f}%         | {ce['mean']:.4f} ± {ce['margin']:.4f}         |")

    print("+------------------+------------------------+------------------------+")

    return results, stats


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(results, stats, save_path=None):
    """Create visualization of repeated-seed results."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['softmax', 'mlp']
    labels = ['Softmax\nRegression', 'MLP\n(h=32)']
    colors = ['steelblue', 'coral']

    # Plot 1: Accuracy with error bars
    ax1 = axes[0]
    means = [stats[m]['accuracy']['mean'] * 100 for m in models]
    errors = [stats[m]['accuracy']['margin'] * 100 for m in models]

    bars1 = ax1.bar(labels, means, yerr=errors, capsize=10, color=colors,
                    edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add individual points
    for i, model in enumerate(models):
        accs = [a * 100 for a in results[model]['accuracy']]
        x = [i] * len(accs)
        ax1.scatter(x, accs, color='black', zorder=5, s=50, alpha=0.7)

    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy (5 seeds, 95% CI)', fontsize=14, fontweight='bold')
    ax1.set_ylim([85, 100])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, err in zip(bars1, means, errors):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.5,
                 f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Cross-Entropy with error bars
    ax2 = axes[1]
    means_ce = [stats[m]['cross_entropy']['mean'] for m in models]
    errors_ce = [stats[m]['cross_entropy']['margin'] for m in models]

    bars2 = ax2.bar(labels, means_ce, yerr=errors_ce, capsize=10, color=colors,
                    edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add individual points
    for i, model in enumerate(models):
        ces = results[model]['cross_entropy']
        x = [i] * len(ces)
        ax2.scatter(x, ces, color='black', zorder=5, s=50, alpha=0.7)

    ax2.set_ylabel('Test Cross-Entropy', fontsize=12)
    ax2.set_title('Test Cross-Entropy (5 seeds, 95% CI)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, err in zip(bars2, means_ce, errors_ce):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                 f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Repeated-Seed Evaluation on Digits Benchmark',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")

    plt.show()

    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Run the evaluation
    results, stats = run_repeated_seed_evaluation(seeds=[0, 1, 2, 3, 4])

    # Create visualization
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_results(results, stats,
                 save_path=figures_dir / 'repeated_seed_evaluation.png')

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
