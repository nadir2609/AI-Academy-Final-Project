from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SoftmaxRegression
from train import train_softmax
from evaluate import evaluate
from plots import plot_decision_boundary, plot_training_dynamics
from pca import PCA

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_synthetic(path: Path):
    data = np.load(path)
    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


def run_synthetic_experiment(
        name: str,
        data_path: Path,
        lr: float = 0.05,
        epochs: int = 200,
        lam: float = 1e-4,
        seed: int = 0,
):
    print(f"\n--- Synthetic experiment: {name} ---")

    X_tr, y_tr, X_v, y_v, X_te, y_te = load_synthetic(data_path)
    d = X_tr.shape[1]
    k = len(np.unique(y_tr))

    model = SoftmaxRegression(d, k, lam=lam, seed=seed)
    history = train_softmax(
        model,
        X_tr,
        y_tr,
        X_v,
        y_v,
        lr=lr,
        epochs=epochs,
        seed=seed,
        verbose=True,
    )

    tag = name.lower().replace(" ", "_")

    plot_decision_boundary(
        model,
        X_tr,
        y_tr,
        title=f"Softmax — {name} (train)",
        save_path=FIG_DIR / f"softmax_{tag}_train_boundary.png",
    )
    plot_decision_boundary(
        model,
        X_te,
        y_te,
        title=f"Softmax — {name} (test)",
        save_path=FIG_DIR / f"softmax_{tag}_test_boundary.png",
    )

    metrics = evaluate(model, X_te, y_te)
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test cross-entropy: {metrics['cross_entropy']:.4f}")
    return model, history, metrics


def run_digits_experiment(
        seeds: list[int] | None = None,
        lr: float = 0.05,
        batch_size: int = 64,
        epochs: int = 200,
        lam: float = 1e-4,
) -> dict:
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    data = np.load(DATA_DIR / "digits_data.npz")
    split = np.load(DATA_DIR / "digits_split_indices.npz")

    X_all, y_all = data["X"], data["y"]
    X_tr = X_all[split["train_idx"]]
    y_tr = y_all[split["train_idx"]]
    X_v = X_all[split["val_idx"]]
    y_v = y_all[split["val_idx"]]
    X_te = X_all[split["test_idx"]]
    y_te = y_all[split["test_idx"]]

    d = X_tr.shape[1]
    k = 10

    print("\n--- Digits benchmark: Softmax Regression ---")
    print(f"  Train={len(y_tr)}, Val={len(y_v)}, Test={len(y_te)}")

    per_seed_results = []
    histories = []

    for seed in seeds:
        model = SoftmaxRegression(d, k, lam=lam, seed=seed)
        hist = train_softmax(
            model,
            X_tr,
            y_tr,
            X_v,
            y_v,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
        )
        res = evaluate(model, X_te, y_te)
        per_seed_results.append(res)
        histories.append(hist)

        print(
            f"  Seed {seed}: test_acc={res['accuracy']:.4f}  "
            f"test_ce={res['cross_entropy']:.4f}  "
            f"best_epoch={hist['best_epoch']}"
        )

    accs = np.array([r["accuracy"] for r in per_seed_results])
    ces = np.array([r["cross_entropy"] for r in per_seed_results])

    t_crit = 2.776
    n_seeds = len(seeds)
    ci_acc = t_crit * accs.std(ddof=1) / np.sqrt(n_seeds)
    ci_ce = t_crit * ces.std(ddof=1) / np.sqrt(n_seeds)

    print(f"\n  ===== Repeated-seed summary (n={n_seeds}) =====")
    print(f"  Test accuracy:      {accs.mean():.4f} ± {ci_acc:.4f}  (95% CI)")
    print(f"  Test cross-entropy: {ces.mean():.4f} ± {ci_ce:.4f}  (95% CI)")
    print("  ================================================")

    plot_training_dynamics(
        histories[0],
        title="Softmax — Digits (seed 0)",
        save_path=FIG_DIR / "softmax_digits_training_dynamics.png",
    )

    return {
        "accs": accs,
        "ces": ces,
        "mean_acc": accs.mean(),
        "ci_acc": ci_acc,
        "mean_ce": ces.mean(),
        "ci_ce": ci_ce,
        "histories": histories,
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_v": X_v,
        "y_v": y_v,
        "X_te": X_te,
        "y_te": y_te,
    }


def run_track_a(digits_results: dict):
    print("\n" + "=" * 60)
    print("TRACK A — PCA / SVD AND INPUT GEOMETRY")
    print("=" * 60)

    X_tr = digits_results["X_tr"]
    y_tr = digits_results["y_tr"]
    X_v = digits_results["X_v"]
    y_v = digits_results["y_v"]
    X_te = digits_results["X_te"]
    y_te = digits_results["y_te"]
    k = 10

    d = X_tr.shape[1]
    pca_full = PCA(n_components=d)
    pca_full.fit(X_tr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ev_ratio = pca_full.explained_variance_ratio_
    cum_ev = np.cumsum(ev_ratio)

    axes[0].bar(range(1, d + 1), ev_ratio, color="steelblue", alpha=0.75)
    axes[0].set_xlabel("Principal component index")
    axes[0].set_ylabel("Fraction of variance explained")
    axes[0].set_title("Scree plot — Digits (individual)")
    axes[0].set_xlim(0.5, d + 0.5)

    axes[1].plot(
        range(1, d + 1),
        cum_ev,
        marker=".",
        markersize=3,
        color="darkorange",
    )
    for m in [10, 20, 40]:
        axes[1].axvline(
            m,
            linestyle="--",
            linewidth=0.8,
            label=f"m={m}: {cum_ev[m - 1] * 100:.1f}%",
        )

    axes[1].set_xlabel("Number of components m")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title("Cumulative explained variance")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "track_a_scree_plot.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/track_a_scree_plot.png")

    for m in [10, 20, 40]:
        print(f"  m={m:2d}: cumulative explained variance = {cum_ev[m - 1] * 100:.2f}%")

    pca2 = PCA(n_components=2)
    pca2.fit(X_tr)
    Z_tr = pca2.transform(X_tr)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.tab10

    for digit in range(10):
        mask = y_tr == digit
        ax.scatter(
            Z_tr[mask, 0],
            Z_tr[mask, 1],
            label=str(digit),
            s=12,
            alpha=0.6,
            color=cmap(digit / 9),
        )

    ax.set_xlabel(f"PC 1 ({ev_ratio[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({ev_ratio[1] * 100:.1f}% var)")
    ax.set_title("2D PCA visualisation — Digits training set")
    ax.legend(title="Digit", ncol=2, fontsize=8, markerscale=2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "track_a_pca2d.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/track_a_pca2d.png")

    pca_dims = [10, 20, 40]
    print(f"\n  Softmax accuracy at fixed PCA dimensions m ∈ {pca_dims}")
    print(f"  {'m':>4}  {'Val acc':>8}  {'Val CE':>8}  {'Test acc':>9}  {'Test CE':>9}")

    track_a_results = {}

    for m in pca_dims:
        pca_m = PCA(n_components=m)
        pca_m.fit(X_tr)

        Xm_tr = pca_m.transform(X_tr)
        Xm_v = pca_m.transform(X_v)
        Xm_te = pca_m.transform(X_te)

        model_m = SoftmaxRegression(m, k, lam=1e-4, seed=0)
        train_softmax(
            model_m,
            Xm_tr,
            y_tr,
            Xm_v,
            y_v,
            lr=0.05,
            batch_size=64,
            epochs=200,
            seed=0,
        )

        val_res = evaluate(model_m, Xm_v, y_v)
        test_res = evaluate(model_m, Xm_te, y_te)
        track_a_results[m] = {"val": val_res, "test": test_res}

        print(
            f"  {m:>4}  {val_res['accuracy']:.4f}    "
            f"{val_res['cross_entropy']:.4f}    "
            f"{test_res['accuracy']:.4f}     "
            f"{test_res['cross_entropy']:.4f}"
        )

    ms = pca_dims
    accs = [track_a_results[m]["test"]["accuracy"] for m in ms]
    ces = [track_a_results[m]["test"]["cross_entropy"] for m in ms]

    labels = [str(m) for m in ms] + ["64 (full)"]
    accs_full = accs + [digits_results["mean_acc"]]
    ces_full = ces + [digits_results["mean_ce"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))

    axes[0].bar(x, accs_full, color=["steelblue"] * 3 + ["darkorange"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("PCA dimension m")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Track A — Test accuracy vs PCA dimension")
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, ces_full, color=["steelblue"] * 3 + ["darkorange"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("PCA dimension m")
    axes[1].set_ylabel("Test cross-entropy")
    axes[1].set_title("Track A — Test cross-entropy vs PCA dimension")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "track_a_pca_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/track_a_pca_comparison.png")

    return track_a_results
