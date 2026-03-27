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