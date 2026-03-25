import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from starter_pack.src.mlp import MLP
from starter_pack.src.train import train
from starter_pack.src.optimizers import SGD, Momentum, Adam


def load_dataset(dataset_name):
    """Load dataset from .npz file."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    data_path = data_dir / f'{dataset_name}.npz'
    data = np.load(data_path)

    # Check if data is already split
    if 'X_train' in data.keys():
        return data['X_train'], data['y_train'], data['X_val'], data['y_val']
    else:
        # Handle digits_data which needs to be split using indices
        X, y = data['X'], data['y']
        # Extract base name (e.g., 'digits' from 'digits_data')
        base_name = dataset_name.replace('_data', '')
        indices_path = data_dir / f'{base_name}_split_indices.npz'
        indices = np.load(indices_path)
        train_idx, val_idx = indices['train_idx'], indices['val_idx']
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def plot_history(history, dataset_name):
    """Plot training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{dataset_name} - Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{dataset_name} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_training_history.png')
    plt.show()


def print_best_params(best_params, dataset_name):
    """Print best model parameters."""
    print(f"\n{'=' * 50}")
    print(f"Best Parameters for {dataset_name}")
    print(f"{'=' * 50}")
    for param_name, param_value in best_params.items():
        print(f"\n{param_name}:")
        print(f"  Shape: {param_value.shape}")
        print(f"  Mean: {np.mean(param_value):.6f}")
        print(f"  Std: {np.std(param_value):.6f}")
        print(f"  Min: {np.min(param_value):.6f}")
        print(f"  Max: {np.max(param_value):.6f}")
    print(f"{'=' * 50}\n")


def main():
    """Main function to train models on all datasets."""
    datasets = ['digits_data', 'linear_gaussian', 'moons']

    for dataset_name in datasets:
        print(f"\n{'#' * 60}")
        print(f"Training on {dataset_name}")
        print(f"{'#' * 60}\n")

        # Load data
        X_train, y_train, X_val, y_val = load_dataset(dataset_name)

        # Determine model parameters based on dataset
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        hidden_dim = 32

        print(f"Dataset info:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Validation samples: {X_val.shape[0]}\n")

        # Create model
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

        # Create optimizer (using Adam as default, can be changed to SGD or Momentum)
        optimizer = Adam()

        # Train model
        best_params, history = train(
            model=model,
            optimizer=optimizer,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=200,
            batch_size=64,
            lam=1e-4
        )

        # Plot training history
        plot_history(history, dataset_name)

        # Print best parameters
        print_best_params(best_params, dataset_name)

        # Print final metrics
        print(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
        print(f"Best Validation Loss: {min(history['val_loss']):.4f}")


if __name__ == '__main__':
    main()
