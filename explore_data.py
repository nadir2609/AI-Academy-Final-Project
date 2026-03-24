import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# STEP 1: Load datasets
print("STEP 1: Loading datasets")

data_dir = Path("starter_pack/data")

# Load Linear Gaussian
linear = np.load(data_dir / "linear_gaussian.npz")
print("\nLinear Gaussian loaded. Keys:", list(linear.keys()))

# Load Moons
moons = np.load(data_dir / "moons.npz")
print("Moons loaded. Keys:", list(moons.keys()))

# Load Digits
digits_raw = np.load(data_dir / "digits_data.npz")
splits = np.load(data_dir / "digits_split_indices.npz")
print("Digits loaded. Keys:", list(digits_raw.keys()))
print("Split indices loaded. Keys:", list(splits.keys()))

# STEP 2: Inspect shapes

print("STEP 2: Dataset Shapes")

print("\nLinear Gaussian:")
print(f"  X_train: {linear['X_train'].shape}")
print(f"  y_train: {linear['y_train'].shape}")
print(f"  X_val: {linear['X_val'].shape}")
print(f"  y_val: {linear['y_val'].shape}")
print(f"  X_test: {linear['X_test'].shape}")
print(f"  y_test: {linear['y_test'].shape}")

print("\nMoons:")
print(f"  X_train: {moons['X_train'].shape}")
print(f"  y_train: {moons['y_train'].shape}")
print(f"  X_val: {moons['X_val'].shape}")
print(f"  y_val: {moons['y_val'].shape}")
print(f"  X_test: {moons['X_test'].shape}")
print(f"  y_test: {moons['y_test'].shape}")

print("\nDigits (raw):")
print(f"  X: {digits_raw['X'].shape} ")
print(f"  y: {digits_raw['y'].shape}")


# STEP 3: Verify splits (60/20/20)

print("\n" + "=" * 60)
print("STEP 3: Verify Train/Val/Test Splits (60/20/20)")
print("=" * 60)


def check_split(dataset_name, y_train, y_val, y_test):
    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)
    n_total = n_train + n_val + n_test

    print(f"\n{dataset_name}:")
    print(f"  Train: {n_train:4d} ({n_train / n_total * 100:5.1f}%)")
    print(f"  Val:   {n_val:4d} ({n_val / n_total * 100:5.1f}%)")
    print(f"  Test:  {n_test:4d} ({n_test / n_total * 100:5.1f}%)")
    print(f"  Total: {n_total:4d}")



check_split("Linear Gaussian", linear["y_train"], linear["y_val"], linear["y_test"])
check_split("Moons", moons["y_train"], moons["y_val"], moons["y_test"])

# For digits, use split indices
digits_y = digits_raw["y"]
check_split(
    "Digits",
    digits_y[splits["train_idx"]],
    digits_y[splits["val_idx"]],
    digits_y[splits["test_idx"]],
)

# STEP 4: Class distributions

print("STEP 4: Class Distributions")


def show_class_distribution(dataset_name, y_train, y_val, y_test):
    print(f"\n{dataset_name}:")
    for split_name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"  {split_name:5s}: Classes {unique.tolist()}, Counts {counts.tolist()}")


show_class_distribution(
    "Linear Gaussian", linear["y_train"], linear["y_val"], linear["y_test"]
)
show_class_distribution("Moons", moons["y_train"], moons["y_val"], moons["y_test"])
show_class_distribution(
    "Digits",
    digits_y[splits["train_idx"]],
    digits_y[splits["val_idx"]],
    digits_y[splits["test_idx"]],
)


# STEP 5: Visualize Linear Gaussian

print("STEP 5: Visualizing Linear Gaussian")


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, split in zip(axes, ["train", "val", "test"]):
    X = linear[f"X_{split}"]
    y = linear[f"y_{split}"]

    for class_id in [0, 1]:
        mask = y == class_id
        ax.scatter(X[mask, 0], X[mask, 1], label=f"Class {class_id}", alpha=0.6, s=40)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{split.capitalize()} Set (n={len(y)})")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Linear Gaussian Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/linear_gaussian_exploration.png", dpi=150, bbox_inches="tight")
plt.show()


# STEP 6: Visualize Moons

print("STEP 6: Visualizing Moons")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, split in zip(axes, ["train", "val", "test"]):
    X = moons[f"X_{split}"]
    y = moons[f"y_{split}"]

    for class_id in [0, 1]:
        mask = y == class_id
        ax.scatter(X[mask, 0], X[mask, 1], label=f"Class {class_id}", alpha=0.6, s=40)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{split.capitalize()} Set (n={len(y)})")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Moons Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/moons_exploration.png", dpi=150, bbox_inches="tight")
plt.show()


# STEP 7: Visualize Digits (one per class)

print("STEP 7: Visualizing Digits ")

X = digits_raw["X"]
y = digits_raw["y"]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for digit in range(10):
    # Find first example of this digit
    idx = np.where(y == digit)[0][0]
    img = X[idx].reshape(8, 8)

    axes[digit].imshow(img, cmap="gray")
    axes[digit].set_title(f"Digit {digit}", fontweight="bold")
    axes[digit].axis("off")

plt.suptitle("Sample Digits (8×8 images)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/digits_samples.png", dpi=150, bbox_inches="tight")
plt.show()


# STEP 8: Visualize digit grid

print("STEP 8: Visualizing 8×8 grid of random digits")

# Get training data
X_train = X[splits["train_idx"]]
y_train = y[splits["train_idx"]]

fig, axes = plt.subplots(8, 8, figsize=(12, 12))

np.random.seed(42)
for i in range(8):
    for j in range(8):
        # Random sample
        idx = np.random.randint(0, len(X_train))
        img = X_train[idx].reshape(8, 8)

        axes[i, j].imshow(img, cmap="gray")
        axes[i, j].axis("off")
        # Add label as title for first row
        if i == 0:
            axes[i, j].set_title(f"{y_train[idx]}", fontsize=10)

plt.suptitle("Random 8×8 Grid of Training Digits", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/digits_grid.png", dpi=150, bbox_inches="tight")
plt.show()

