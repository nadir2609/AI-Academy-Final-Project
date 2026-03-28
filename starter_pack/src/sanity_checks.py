import numpy as np

from utils import softmax, cross_entropy_loss, one_hot
from model import SoftmaxRegression
from train import train_softmax


def run_sanity_checks():
    """
    Implementation sanity checks.
    """
    print("\n" + "=" * 60)
    print("IMPLEMENTATION SANITY CHECKS")
    print("=" * 60)

    rng = np.random.default_rng(42)
    d, k = 4, 3

    S_test = rng.standard_normal((10, k))
    P_test = softmax(S_test)
    row_sums = P_test.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12), "Softmax rows do not sum to 1!"
    print(
        f"[PASS] Check 1 — Softmax row sums: min={row_sums.min():.6f}, "
        f"max={row_sums.max():.6f}"
    )

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
    print(f"[PASS] Check 2 — Loss decrease: {L0:.4f} → {L1:.4f}")

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
    print(f"[PASS] Check 3 — Tiny overfit: loss={Lf:.4f}, accuracy={acc_f:.2f}")

    eps = 1e-5
    model_g = SoftmaxRegression(d, k, lam=1e-4, seed=2)
    Xg = rng.standard_normal((6, d))
    yg = rng.integers(0, k, size=6)
    Yg = one_hot(yg, k)

    _, Pg = model_g.forward(Xg)
    dW_anal, _ = model_g.backward(Xg, Pg, Yg)

    dW_num = np.zeros_like(model_g.W)
    W_orig = model_g.W.copy()

    for i in range(k):
        for j in range(d):
            model_g.W = W_orig.copy()
            model_g.W[i, j] += eps
            _, Pp = model_g.forward(Xg)
            Lp = cross_entropy_loss(Pp, Yg, model_g.W, model_g.lam)

            model_g.W = W_orig.copy()
            model_g.W[i, j] -= eps
            _, Pm = model_g.forward(Xg)
            Lm = cross_entropy_loss(Pm, Yg, model_g.W, model_g.lam)

            dW_num[i, j] = (Lp - Lm) / (2 * eps)

    model_g.W = W_orig

    rel_err = np.abs(dW_anal - dW_num) / (np.abs(dW_anal) + np.abs(dW_num) + 1e-12)
    max_rel = rel_err.max()
    assert max_rel < 1e-4, f"Gradient check failed: max relative error = {max_rel:.2e}"
    print(f"[PASS] Check 4 — Gradient check: max relative error = {max_rel:.2e}")

    X_rand = rng.standard_normal((50, d))
    y_rand = rng.integers(0, k, size=50)
    model_r = SoftmaxRegression(d, k, lam=1e-4, seed=3)
    train_softmax(
        model_r,
        X_rand,
        y_rand,
        X_rand,
        y_rand,
        lr=0.05,
        batch_size=16,
        epochs=20,
        seed=3,
    )

    has_bad = np.any(np.isnan(model_r.W)) or np.any(np.isinf(model_r.W))
    assert not has_bad, "NaN or Inf detected in weights!"
    print("[PASS] Check 5 — No NaN/Inf in weights after training")

    print("=" * 60)
    print("All sanity checks passed.\n")
