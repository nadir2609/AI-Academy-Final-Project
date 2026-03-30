"""
Training module for neural network models.

Contains training loops for:
  - Softmax Regression with mini-batch SGD
  - MLP (Multi-Layer Perceptron) with various optimizers
"""

import numpy as np
import copy
from typing import Dict, Tuple, Any
from utils import one_hot, cross_entropy_loss


# =============================================================================
# SOFTMAX REGRESSION TRAINING
# =============================================================================

def train_softmax(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 0.05,
    batch_size: int = 64,
    epochs: int = 200,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Train softmax regression model using mini-batch SGD.

    Training Strategy:
      1. Shuffle training data each epoch
      2. Process in mini-batches
      3. Update parameters using gradient descent
      4. Track validation loss for early stopping
      5. Save best model parameters

    Args:
        model: SoftmaxRegression model instance with forward(), backward(), step() methods
        X_train: Training features (n_train, input_dim)
        y_train: Training labels (n_train,)
        X_val: Validation features (n_val, input_dim)
        y_val: Validation labels (n_val,)
        lr: Learning rate for SGD (default: 0.05)
        batch_size: Mini-batch size (default: 64)
        epochs: Number of training epochs (default: 200)
        seed: Random seed for reproducibility (default: 0)
        verbose: Print progress every 50 epochs (default: False)

    Returns:
        Dictionary containing:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'val_acc': List of validation accuracies per epoch
            - 'best_epoch': Epoch with lowest validation loss
    """
    # Get model parameters
    k = model.k  # Number of classes
    n = X_train.shape[0]  # Number of training samples
    rng = np.random.default_rng(seed)

    # Initialize history tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Track best model checkpoint
    best_val_loss = np.inf
    best_W = model.W.copy()
    best_b = model.b.copy()
    best_epoch = 0

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    for epoch in range(epochs):
        # Shuffle training data
        indices = rng.permutation(n)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch SGD
        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            y_onehot = one_hot(y_batch, k)
            _, probs_batch = model.forward(X_batch)

            # Backward pass
            dW, db = model.backward(X_batch, probs_batch, y_onehot)

            # Parameter update
            model.step(dW, db, lr)

        # =====================================================================
        # EPOCH EVALUATION
        # =====================================================================
        # Compute training loss
        _, probs_train = model.forward(X_train)
        y_train_onehot = one_hot(y_train, k)
        train_loss = cross_entropy_loss(probs_train, y_train_onehot, model.W, model.lam)

        # Compute validation loss and accuracy
        _, probs_val = model.forward(X_val)
        y_val_onehot = one_hot(y_val, k)
        val_loss = cross_entropy_loss(probs_val, y_val_onehot, model.W, model.lam)
        val_acc = np.mean(np.argmax(probs_val, axis=1) == y_val)

        # Record metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpoint best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_W = model.W.copy()
            best_b = model.b.copy()
            best_epoch = epoch

        # Print progress
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(
                f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

    # =========================================================================
    # RESTORE BEST MODEL
    # =========================================================================
    model.W = best_W
    model.b = best_b
    history["best_epoch"] = best_epoch

    return history


# =============================================================================
# MLP NEURAL NETWORK TRAINING
# =============================================================================

# Training constants
_EPSILON = 1e-8  # For numerical stability in log operations


def train(
    model,
    optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lam: float = 1e-4
) -> Tuple[Dict[str, np.ndarray], Dict[str, list], int]:
    """
    Train MLP model using mini-batch SGD with various optimizers.

    Training Strategy:
      1. Mini-batch gradient descent with user-specified optimizer
      2. L2 regularization on weight matrices
      3. Validation-based early stopping
      4. Save best model by validation loss

    Args:
        model: MLP instance with forward(), backward() methods
        optimizer: Optimizer instance (SGD, Momentum, or Adam) with step() method
        X_train: Training features (n_train, input_dim)
        y_train: Training labels (n_train,)
        X_val: Validation features (n_val, input_dim)
        y_val: Validation labels (n_val,)
        epochs: Number of training epochs (default: 200)
        batch_size: Mini-batch size (default: 64)
        lam: L2 regularization coefficient (default: 1e-4)

    Returns:
        best_params: Dictionary with best model parameters {'W1', 'b1', 'W2', 'b2'}
        history: Dictionary with training history:
            - 'train_loss': List of training losses per epoch
            - 'train_acc': List of training accuracies per epoch
            - 'val_loss': List of validation losses per epoch
            - 'val_acc': List of validation accuracies per epoch
        best_epoch: Epoch index with lowest validation loss
    """
    num_train = X_train.shape[0]
    min_val_loss = float('inf')

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def pack_params() -> Dict[str, np.ndarray]:
        """Pack model parameters into a dictionary."""
        return {
            'W1': model.W1,
            'b1': model.b1,
            'W2': model.W2,
            'b2': model.b2
        }

    def compute_l2_penalty() -> float:
        """Compute L2 regularization penalty on weight matrices."""
        return 0.5 * lam * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2))

    def compute_loss_and_accuracy(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Compute loss and accuracy on given dataset.

        Args:
            X: Input features
            y: True labels

        Returns:
            loss: Cross-entropy loss + L2 penalty
            accuracy: Fraction of correct predictions
        """
        # Forward pass
        probs = model.forward(X)

        # Cross-entropy loss (with numerical stability)
        data_loss = -np.mean(np.log(probs[np.arange(y.shape[0]), y] + _EPSILON))

        # Total loss = cross-entropy + L2 regularization
        loss = data_loss + compute_l2_penalty()

        # Accuracy
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)

        return loss, accuracy

    # =========================================================================
    # TRAINING INITIALIZATION
    # =========================================================================
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_params = pack_params()
    best_epoch = 0

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    for epoch in range(epochs):

        # =====================================================================
        # SHUFFLE AND MINI-BATCH TRAINING
        # =====================================================================
        # Shuffle training data
        shuffled_indices = np.random.permutation(num_train)
        X_train_shuffled = X_train[shuffled_indices]
        y_train_shuffled = y_train[shuffled_indices]

        # Process mini-batches
        for start in range(0, num_train, batch_size):
            end = start + batch_size

            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward pass
            model.forward(X_batch)

            # Backward pass
            grads = model.backward(y_batch)

            # Add L2 regularization gradient
            grads['W1'] += lam * model.W1
            grads['W2'] += lam * model.W2

            # Optimizer step
            params = pack_params()
            optimizer.step(params, grads)

            # Update model parameters
            model.W1, model.b1 = params['W1'], params['b1']
            model.W2, model.b2 = params['W2'], params['b2']

        # =====================================================================
        # EPOCH EVALUATION
        # =====================================================================
        train_loss, train_acc = compute_loss_and_accuracy(X_train, y_train)
        val_loss, val_acc = compute_loss_and_accuracy(X_val, y_val)

        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Checkpoint best model (by validation loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_params = copy.deepcopy(pack_params())
            best_epoch = epoch

    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    return best_params, history, best_epoch
