"""
Math4AI Capstone — Softmax Regression with Track A (PCA/SVD)
=======================================================================
Implements:
  - Numerically stable softmax and cross-entropy
  - Mini-batch SGD with L2 regularisation
  - Best-checkpoint selection on validation cross-entropy (digits)
  - Decision-boundary plots for synthetic tasks
  - Repeated-seed evaluation (5 seeds) for the digits benchmark
  - Track A: scree plot, 2D PCA visualisation, softmax at m ∈ {10, 20, 40}
  - Implementation sanity checks section

"""

from __future__ import annotations


import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("starter_pack/data")
FIG_DIR = Path("starter_pack/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Core math utilities
# ---------------------------------------------------------------------------

def softmax(S: np.ndarray) -> np.ndarray:
    """
    Numerically stable row-wise softmax.
    """
    shift = S - S.max(axis=1, keepdims=True)   # broadcast over columns
    E = np.exp(shift)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy_loss(P: np.ndarray, Y_onehot: np.ndarray,
                       W: np.ndarray, lam: float) -> float:
    """
    Mean cross-entropy loss with L2 regularisation.
    """
    n = P.shape[0]
    log_P = np.log(np.clip(P, 1e-15, 1.0))
    ce = -np.sum(Y_onehot * log_P) / n
    reg = (lam / 2.0) * np.sum(W ** 2)
    return ce + reg


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    """Convert integer label vector to one-hot matrix of shape (n, k)."""
    n = len(y)
    Y = np.zeros((n, k), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    return Y


# ---------------------------------------------------------------------------
# 2. Softmax regression: forward, backward, update
# ---------------------------------------------------------------------------

class SoftmaxRegression:
    """
    Multiclass softmax regression.

    Model:  s(x) = W x + b,   P = softmax(S)
    Loss:   cross-entropy + L2 regularisation on W

    Parameters
    ----------
    d : input dimension
    k : number of classes
    lam : L2 regularisation coefficient
    seed : random seed for weight initialisation
    """

    def __init__(self, d: int, k: int, lam: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Xavier-style initialisation: keeps gradients well-scaled at start
        scale = np.sqrt(2.0 / (d + k))
        self.W = rng.standard_normal((k, d)) * scale   # shape (k, d)
        self.b = np.zeros(k)                            # shape (k,)
        self.lam = lam
        self.k = k

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray):
        """
        Compute logits and probabilities for a batch X of shape (n, d).

        Returns
        -------
        S : np.ndarray, shape (n, k)  — logits
        P : np.ndarray, shape (n, k)  — probabilities
        """
        S = X @ self.W.T + self.b      # (n, k)
        P = softmax(S)                 # (n, k)
        return S, P

    # ------------------------------------------------------------------
    # Backward pass  
    # ------------------------------------------------------------------
    def backward(self, X: np.ndarray, P: np.ndarray, Y_onehot: np.ndarray):
        """
        Compute parameter gradients via the chain rule.

        Starting from the average cross-entropy:
            dL/dS = (P - Y) / n               (well-known softmax-CE result)
            dL/dW = (dL/dS)^T @ X  + lambda*W  (chain rule + L2 term)
            dL/db = (dL/dS)^T @ 1_n

        Returns
        -------
        dW : gradient w.r.t. W, shape (k, d)
        db : gradient w.r.t. b, shape (k,)
        """
        n = X.shape[0]
        dS = (P - Y_onehot) / n       # (n, k)
        dW = dS.T @ X + self.lam * self.W   # (k, d)
        db = dS.sum(axis=0)           # (k,)
        return dW, db

    # ------------------------------------------------------------------
    # SGD update
    # ------------------------------------------------------------------
    def step(self, dW: np.ndarray, db: np.ndarray, lr: float):
        self.W -= lr * dW
        self.b -= lr * db

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        _, P = self.forward(X)
        return np.argmax(P, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, P = self.forward(X)
        return P


# ---------------------------------------------------------------------------
# 3. Training loop with best-checkpoint tracking
# ---------------------------------------------------------------------------

def train_softmax(
    model: SoftmaxRegression,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    lr: float = 0.05,
    batch_size: int = 64,
    epochs: int = 200,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Mini-batch SGD training loop.

    Returns a history dict with train/val loss and val accuracy per epoch,
    and the best weights (lowest validation cross-entropy checkpoint).
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
        # ---- shuffle ----
        idx = rng.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        # ---- mini-batch SGD ----
        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            Yb = one_hot(yb, k)
            _, Pb = model.forward(Xb)
            dW, db = model.backward(Xb, Pb, Yb)
            model.step(dW, db, lr)

        # ---- epoch metrics ----
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

        # ---- checkpoint ----
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_W = model.W.copy()
            best_b = model.b.copy()
            best_epoch = epoch

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={tr_loss:.4f}  "
                  f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

    # restore best checkpoint
    model.W = best_W
    model.b = best_b
    history["best_epoch"] = best_epoch
    return history


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model: SoftmaxRegression, X: np.ndarray, y: np.ndarray) -> dict:
    """Return accuracy and mean cross-entropy on a split."""
    k = model.k
    P = model.predict_proba(X)
    Y = one_hot(y, k)
    acc = np.mean(np.argmax(P, axis=1) == y)
    ce = -np.sum(Y * np.log(np.clip(P, 1e-15, 1.0))) / len(y)
    return {"accuracy": acc, "cross_entropy": ce}


# ---------------------------------------------------------------------------
# 5. Decision-boundary plotting
# ---------------------------------------------------------------------------

def plot_decision_boundary(
    model: SoftmaxRegression,
    X: np.ndarray, y: np.ndarray,
    title: str,
    save_path: Path,
    transform=None,
):
   
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if transform is not None:
        grid = transform(grid)
    Z = model.predict(grid).reshape(xx.shape)

    cmap_bg = plt.cm.RdYlBu
    cmap_pts = plt.cm.Dark2

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], label=f"Class {cls}",
                   s=25, edgecolors="k", linewidths=0.4,
                   color=cmap_pts(cls / max(np.unique(y).max(), 1)))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 6. Training-dynamics plot
# ---------------------------------------------------------------------------

def plot_training_dynamics(history: dict, title: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss")
    axes[0].axvline(history.get("best_epoch", 0) + 1,
                    color="gray", linestyle="--", linewidth=0.8, label="Best epoch")
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


# ---------------------------------------------------------------------------
# 7. Sanity checks  
# ---------------------------------------------------------------------------

def run_sanity_checks():
    """
    Implementation Sanity Checks.

    Checks performed
    ----------------
    1. Softmax probabilities sum to 1.
    2. Loss decreases on a tiny subset after a few SGD steps.
    3. Model can overfit a tiny 5-sample dataset (near-zero loss).
    4. Numerical gradient check: finite differences vs analytical gradient.
    5. No NaNs or Infs produced during training on random data.
    """
    print("\n" + "="*60)
    print("IMPLEMENTATION SANITY CHECKS")
    print("="*60)

    rng = np.random.default_rng(42)
    d, k = 4, 3

    # ---- Check 1: softmax sums to 1 ----
    S_test = rng.standard_normal((10, k))
    P_test = softmax(S_test)
    row_sums = P_test.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12), "Softmax rows do not sum to 1!"
    print(f"[PASS] Check 1 — Softmax row sums: min={row_sums.min():.6f}, "
          f"max={row_sums.max():.6f} (should be ≈ 1)")

    # ---- Check 2: loss decreases on a tiny batch ----
    Xc = rng.standard_normal((8, d))
    yc = rng.integers(0, k, size=8)
    model_c = SoftmaxRegression(d, k, lam=0.0, seed=0)
    _, P0 = model_c.forward(Xc)
    L0 = cross_entropy_loss(P0, one_hot(yc, k), model_c.W, lam=0.0)
    for _ in range(50):
        _, Pb = model_c.forward(Xc)
        dW, db = model_c.backward(Xc, Pb, one_hot(yc, k))
        model_c.step(dW, db, lr=0.1)
    _, P1 = model_c.forward(Xc)
    L1 = cross_entropy_loss(P1, one_hot(yc, k), model_c.W, lam=0.0)
    assert L1 < L0, f"Loss did not decrease: {L0:.4f} -> {L1:.4f}"
    print(f"[PASS] Check 2 — Loss decrease: {L0:.4f} → {L1:.4f} after 50 steps")

    # ---- Check 3: overfit tiny dataset ----
    X_tiny = rng.standard_normal((5, d))
    y_tiny = np.arange(5) % k
    model_t = SoftmaxRegression(d, k, lam=0.0, seed=1)
    for _ in range(2000):
        _, Pt = model_t.forward(X_tiny)
        dW, db = model_t.backward(X_tiny, Pt, one_hot(y_tiny, k))
        model_t.step(dW, db, lr=0.3)
    _, Pf = model_t.forward(X_tiny)
    Lf = cross_entropy_loss(Pf, one_hot(y_tiny, k), model_t.W, lam=0.0)
    acc_f = np.mean(np.argmax(Pf, axis=1) == y_tiny)
    print(f"[PASS] Check 3 — Tiny overfit: loss={Lf:.4f}, accuracy={acc_f:.2f} "
          f"(expect loss ≈ 0, acc ≈ 1)")

    # ---- Check 4: numerical gradient check ----
    eps = 1e-5
    model_g = SoftmaxRegression(d, k, lam=1e-4, seed=2)
    Xg = rng.standard_normal((6, d))
    yg = rng.integers(0, k, size=6)
    Yg = one_hot(yg, k)

    _, Pg = model_g.forward(Xg)
    dW_anal, _ = model_g.backward(Xg, Pg, Yg)

    # Finite-difference gradient for a random subset of W entries
    dW_num = np.zeros_like(model_g.W)
    for i in range(k):
        for j in range(d):
            W_plus = model_g.W.copy(); W_plus[i, j] += eps
            model_g.W = W_plus
            _, Pp = model_g.forward(Xg)
            Lp = cross_entropy_loss(Pp, Yg, model_g.W, model_g.lam)

            W_minus = model_g.W.copy(); W_minus[i, j] -= 2 * eps
            model_g.W = W_minus
            _, Pm = model_g.forward(Xg)
            Lm = cross_entropy_loss(Pm, Yg, model_g.W, model_g.lam)

            model_g.W[i, j] += eps   # restore
            dW_num[i, j] = (Lp - Lm) / (2 * eps)

    rel_err = np.abs(dW_anal - dW_num) / (np.abs(dW_anal) + np.abs(dW_num) + 1e-12)
    max_rel = rel_err.max()
    assert max_rel < 1e-4, f"Gradient check failed: max relative error = {max_rel:.2e}"
    print(f"[PASS] Check 4 — Gradient check: max relative error = {max_rel:.2e} "
          f"(should be < 1e-4)")

    # ---- Check 5: no NaN or Inf during random training ----
    X_rand = rng.standard_normal((50, d))
    y_rand = rng.integers(0, k, size=50)
    model_r = SoftmaxRegression(d, k, lam=1e-4, seed=3)
    train_softmax(model_r, X_rand, y_rand, X_rand, y_rand,
                  lr=0.05, batch_size=16, epochs=20, seed=3)
    has_nan = np.any(np.isnan(model_r.W)) or np.any(np.isinf(model_r.W))
    assert not has_nan, "NaN or Inf detected in weights!"
    print(f"[PASS] Check 5 — No NaN/Inf in weights after training")

    print("="*60)
    print("All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# 8. Experiment helpers
# ---------------------------------------------------------------------------

def load_synthetic(path: Path):
    data = np.load(path)
    return (data["X_train"], data["y_train"],
            data["X_val"],   data["y_val"],
            data["X_test"],  data["y_test"])


def run_synthetic_experiment(name: str, data_path: Path,
                             lr: float = 0.05, epochs: int = 200,
                             lam: float = 1e-4, seed: int = 0):
    """Train softmax on a synthetic dataset and plot decision boundaries."""
    print(f"\n--- Synthetic experiment: {name} ---")
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_synthetic(data_path)
    d = X_tr.shape[1]
    k = len(np.unique(y_tr))

    model = SoftmaxRegression(d, k, lam=lam, seed=seed)
    history = train_softmax(model, X_tr, y_tr, X_v, y_v,
                             lr=lr, epochs=epochs, seed=seed, verbose=True)

    # Decision boundary on training split
    tag = name.lower().replace(" ", "_")
    plot_decision_boundary(model, X_tr, y_tr,
                           title=f"Softmax — {name} (train)",
                           save_path=FIG_DIR / f"softmax_{tag}_train_boundary.png")
    plot_decision_boundary(model, X_te, y_te,
                           title=f"Softmax — {name} (test)",
                           save_path=FIG_DIR / f"softmax_{tag}_test_boundary.png")

    metrics = evaluate(model, X_te, y_te)
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test cross-entropy: {metrics['cross_entropy']:.4f}")
    return model, history, metrics


# ---------------------------------------------------------------------------
# 9. Digits benchmark with repeated-seed evaluation
# ---------------------------------------------------------------------------

def run_digits_experiment(seeds: list[int] = None,
                          lr: float = 0.05, batch_size: int = 64,
                          epochs: int = 200, lam: float = 1e-4,
                          hidden_width: int = 32) -> dict:
    """
    Train softmax on the digits benchmark with repeated-seed evaluation.

    Returns dict of per-seed results and aggregate statistics.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    data = np.load(DATA_DIR / "digits_data.npz")
    split = np.load(DATA_DIR / "digits_split_indices.npz")
    X_all, y_all = data["X"], data["y"]
    X_tr  = X_all[split["train_idx"]]
    y_tr  = y_all[split["train_idx"]]
    X_v   = X_all[split["val_idx"]]
    y_v   = y_all[split["val_idx"]]
    X_te  = X_all[split["test_idx"]]
    y_te  = y_all[split["test_idx"]]

    d = X_tr.shape[1]  # 64
    k = 10

    print("\n--- Digits benchmark: Softmax Regression ---")
    print(f"  Train={len(y_tr)}, Val={len(y_v)}, Test={len(y_te)}")

    per_seed_results = []
    histories = []

    for seed in seeds:
        model = SoftmaxRegression(d, k, lam=lam, seed=seed)
        hist = train_softmax(model, X_tr, y_tr, X_v, y_v,
                              lr=lr, batch_size=batch_size,
                              epochs=epochs, seed=seed)
        res = evaluate(model, X_te, y_te)
        per_seed_results.append(res)
        histories.append(hist)
        print(f"  Seed {seed}: test_acc={res['accuracy']:.4f}  "
              f"test_ce={res['cross_entropy']:.4f}  "
              f"best_epoch={hist['best_epoch']}")

    accs = np.array([r["accuracy"] for r in per_seed_results])
    ces  = np.array([r["cross_entropy"] for r in per_seed_results])

    # 95% CI using t_{0.025, 4} = 2.776 (n=5 seeds)
    t_crit = 2.776
    n_seeds = len(seeds)
    ci_acc = t_crit * accs.std() / np.sqrt(n_seeds)
    ci_ce  = t_crit * ces.std()  / np.sqrt(n_seeds)

    print(f"\n  ===== Repeated-seed summary (n={n_seeds}) =====")
    print(f"  Test accuracy:     {accs.mean():.4f} ± {ci_acc:.4f}  (95% CI)")
    print(f"  Test cross-entropy:{ces.mean():.4f} ± {ci_ce:.4f}  (95% CI)")
    print(f"  ================================================")

    # Plot training dynamics for seed 0 (representative)
    plot_training_dynamics(
        histories[0],
        title="Softmax — Digits (seed 0)",
        save_path=FIG_DIR / "softmax_digits_training_dynamics.png"
    )

    return {
        "accs": accs, "ces": ces,
        "mean_acc": accs.mean(), "ci_acc": ci_acc,
        "mean_ce": ces.mean(),  "ci_ce": ci_ce,
        "histories": histories,
        "X_tr": X_tr, "y_tr": y_tr,
        "X_v": X_v,   "y_v": y_v,
        "X_te": X_te, "y_te": y_te,
    }


# ---------------------------------------------------------------------------
# 10. Track A — PCA / SVD analysis
# ---------------------------------------------------------------------------

class PCA:
    """
    Principal Component Analysis via SVD.

    Uses numpy's np.linalg.svd (permitted by §5.1 for Track A).
    Centres the data before decomposing.

    The covariance structure satisfies:
        X_c = U S V^T   (economy SVD)
    Principal components are the rows of V^T (right singular vectors).
    Explained variance of component i ∝ σ_i^2.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None   # shape (n_components, d)
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        # Economy SVD: X_c = U @ diag(s) @ Vt
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        # Variance explained by each component
        total_var = (s ** 2).sum()
        self.explained_variance_ = s[:self.n_components] ** 2 / (len(X) - 1)
        self.explained_variance_ratio_ = s[:self.n_components] ** 2 / total_var
        self._all_s = s
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the first n_components principal axes."""
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def run_track_a(digits_results: dict):
    """
    Track A: PCA/SVD and input geometry — required work:
      1. Scree plot
      2. 2D PCA visualisation of digits
      3. Softmax at m ∈ {10, 20, 40}
      4. Interpretation text printed to stdout
    """
    print("\n" + "="*60)
    print("TRACK A — PCA / SVD AND INPUT GEOMETRY")
    print("="*60)

    X_tr = digits_results["X_tr"]
    y_tr = digits_results["y_tr"]
    X_v  = digits_results["X_v"]
    y_v  = digits_results["y_v"]
    X_te = digits_results["X_te"]
    y_te = digits_results["y_te"]
    k = 10

    # ------------------------------------------------------------------
    # Fit full PCA on training data only (avoid data leakage)
    # ------------------------------------------------------------------
    d = X_tr.shape[1]   # 64
    pca_full = PCA(n_components=d)
    pca_full.fit(X_tr)

    # ------------------------------------------------------------------
    # A1. Scree plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ev_ratio = pca_full.explained_variance_ratio_
    cum_ev   = np.cumsum(ev_ratio)

    axes[0].bar(range(1, d + 1), ev_ratio, color="steelblue", alpha=0.75)
    axes[0].set_xlabel("Principal component index")
    axes[0].set_ylabel("Fraction of variance explained")
    axes[0].set_title("Scree plot — Digits (individual)")
    axes[0].set_xlim(0.5, d + 0.5)

    axes[1].plot(range(1, d + 1), cum_ev, marker=".", markersize=3,
                 color="darkorange")
    for m in [10, 20, 40]:
        axes[1].axvline(m, linestyle="--", linewidth=0.8,
                        label=f"m={m}: {cum_ev[m-1]*100:.1f}%")
    axes[1].set_xlabel("Number of components m")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title("Cumulative explained variance")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "track_a_scree_plot.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/track_a_scree_plot.png")

    for m in [10, 20, 40]:
        print(f"  m={m:2d}: cumulative explained variance = "
              f"{cum_ev[m-1]*100:.2f}%")

    # ------------------------------------------------------------------
    # A2. 2D PCA visualisation
    # ------------------------------------------------------------------
    pca2 = PCA(n_components=2)
    pca2.fit(X_tr)
    Z_tr = pca2.transform(X_tr)
    Z_te = pca2.transform(X_te)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.tab10
    for digit in range(10):
        mask = y_tr == digit
        ax.scatter(Z_tr[mask, 0], Z_tr[mask, 1],
                   label=str(digit), s=12, alpha=0.6,
                   color=cmap(digit / 9))
    ax.set_xlabel(f"PC 1 ({ev_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({ev_ratio[1]*100:.1f}% var)")
    ax.set_title("2D PCA visualisation — Digits training set")
    ax.legend(title="Digit", ncol=2, fontsize=8, markerscale=2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "track_a_pca2d.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/track_a_pca2d.png")

    # ------------------------------------------------------------------
    # A3. Softmax at m ∈ {10, 20, 40}
    # ------------------------------------------------------------------
    pca_dims = [10, 20, 40]
    print(f"\n  Softmax accuracy at fixed PCA dimensions m ∈ {pca_dims}")
    print(f"  {'m':>4}  {'Val acc':>8}  {'Val CE':>8}  "
          f"{'Test acc':>9}  {'Test CE':>9}")

    track_a_results = {}
    for m in pca_dims:
        pca_m = PCA(n_components=m)
        pca_m.fit(X_tr)
        Xm_tr = pca_m.transform(X_tr)
        Xm_v  = pca_m.transform(X_v)
        Xm_te = pca_m.transform(X_te)

        model_m = SoftmaxRegression(m, k, lam=1e-4, seed=0)
        train_softmax(model_m, Xm_tr, y_tr, Xm_v, y_v,
                      lr=0.05, batch_size=64, epochs=200, seed=0)

        val_res  = evaluate(model_m, Xm_v,  y_v)
        test_res = evaluate(model_m, Xm_te, y_te)
        track_a_results[m] = {"val": val_res, "test": test_res}

        print(f"  {m:>4}  {val_res['accuracy']:.4f}    "
              f"{val_res['cross_entropy']:.4f}    "
              f"{test_res['accuracy']:.4f}     "
              f"{test_res['cross_entropy']:.4f}")

    # ------------------------------------------------------------------
    # A4. Summary bar plot for Track A
    # ------------------------------------------------------------------
    ms   = pca_dims
    accs = [track_a_results[m]["test"]["accuracy"] for m in ms]
    ces  = [track_a_results[m]["test"]["cross_entropy"] for m in ms]

    # Also include full-dimension baseline from repeated-seed mean
    ms_full   = ms + [64]
    accs_full = accs + [digits_results["mean_acc"]]
    ces_full  = ces  + [digits_results["mean_ce"]]
    labels    = [str(m) for m in ms] + ["64 (full)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))

    axes[0].bar(x, accs_full, color=["steelblue"]*3 + ["darkorange"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("PCA dimension m")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Track A — Test accuracy vs PCA dimension")
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, ces_full, color=["steelblue"]*3 + ["darkorange"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("PCA dimension m")
    axes[1].set_ylabel("Test cross-entropy")
    axes[1].set_title("Track A — Test cross-entropy vs PCA dimension")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "track_a_pca_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/track_a_pca_comparison.png")


    return track_a_results


# ---------------------------------------------------------------------------
# 11. Main orchestration
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # ---- Sanity checks ----
    run_sanity_checks()

    # ---- Synthetic: Linear Gaussian ----
    run_synthetic_experiment(
        "Linear Gaussian",
        DATA_DIR / "linear_gaussian.npz",
        lr=0.05, epochs=200, lam=1e-4, seed=0
    )

    # ---- Synthetic: Moons ----
    run_synthetic_experiment(
        "Moons",
        DATA_DIR / "moons.npz",
        lr=0.05, epochs=200, lam=1e-4, seed=0
    )

    # ---- Digits benchmark with 5-seed repeated evaluation ----
    digits_results = run_digits_experiment(
        seeds=[0, 1, 2, 3, 4],
        lr=0.05, batch_size=64, epochs=200, lam=1e-4
    )

    # ---- Track A: PCA/SVD ----
    run_track_a(digits_results)

    print(f"\nAll done in {time.time() - t0:.1f}s.  "
          f"Figures saved to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
