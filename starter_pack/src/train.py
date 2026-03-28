import numpy as np

from .utils import one_hot, cross_entropy_loss


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
) -> dict:
    """
    Mini-batch SGD training loop.
    """
    k = model.k
    n = X_train.shape[0]
    rng = np.random.default_rng(seed)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = np.inf
    best_W = model.W.copy()
    best_b = model.b.copy()
    best_epoch = 0

    for epoch in range(epochs):
        idx = rng.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]

            Yb = one_hot(yb, k)
            _, Pb = model.forward(Xb)
            dW, db = model.backward(Xb, Pb, Yb)
            model.step(dW, db, lr)

        _, P_tr = model.forward(X_train)
        Y_tr = one_hot(y_train, k)
        tr_loss = cross_entropy_loss(P_tr, Y_tr, model.W, model.lam)

        _, P_v = model.forward(X_val)
        Y_v = one_hot(y_val, k)
        v_loss = cross_entropy_loss(P_v, Y_v, model.W, model.lam)
        v_acc = np.mean(np.argmax(P_v, axis=1) == y_val)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_W = model.W.copy()
            best_b = model.b.copy()
            best_epoch = epoch

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(
                f"  epoch {epoch:3d}  train_loss={tr_loss:.4f}  "
                f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}"
            )

    model.W = best_W
    model.b = best_b
    history["best_epoch"] = best_epoch
    return history
import copy

import numpy as np


def train(model, optimizer, X_train, y_train, X_val, y_val, epochs=200, batch_size=64, lam=1e-4):
    """
    Train the MLP model.

    Args:
        model: MLP instance
        optimizer: Optimizer instance (SGD, Momentum, or Adam)
        X_train: (n_train, d) training data
        y_train: (n_train,) training labels
        X_val: (n_val, d) validation data
        y_val: (n_val,) validation labels
        epochs: number of training epochs
        batch_size: mini-batch size
        lam: L2 regularization coefficient

    Returns:
        history: dict with 'train_loss', 'train_acc', 'val_loss', 'val_acc' lists
        best_params: dict with best model parameters
    """
    eps = 1e-8
    num_train = X_train.shape[0]
    min_val_loss = float('inf')

    def pack_params():
        return {'W1': model.W1, 'b1': model.b1, 'W2': model.W2, 'b2': model.b2}

    # L2 regularization term(prevent overfitting)
    def l2_penalty():
        return 0.5 * lam * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2))

    # compute loss and accuracy for a given dataset
    def compute_loss_and_accuracy(X, y):
        probs = model.forward(X)
        data_loss = -np.mean(np.log(probs[np.arange(y.shape[0]), y] + eps))
        loss = data_loss + l2_penalty()
        accuracy = np.mean(np.argmax(probs, axis=1) == y) # accuracy is the fraction of correct predictions
        return loss, accuracy

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_params = pack_params()
    best_epoch = 0

    for epoch in range(epochs):

        # shuffle data
        shuffled_indices = np.random.permutation(num_train)
        X_train_shuffled = X_train[shuffled_indices]
        y_train_shuffled = y_train[shuffled_indices]

        for start in range(0, num_train, batch_size):
            end = start + batch_size

            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            model.forward(X_batch)
            grads = model.backward(y_batch)

            grads['W1'] += lam * model.W1
            grads['W2'] += lam * model.W2

            params = pack_params()

            # update parameters
            optimizer.step(params, grads)
            model.W1, model.b1, model.W2, model.b2 = (
                params['W1'], params['b1'], params['W2'], params['b2']
            )

        train_loss, train_acc = compute_loss_and_accuracy(X_train, y_train)
        val_loss, val_acc = compute_loss_and_accuracy(X_val, y_val)

        # save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model parameters based on validation loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_params = copy.deepcopy(pack_params())  # deep copy for params
            best_epoch = epoch

    return best_params, history,best_epoch
