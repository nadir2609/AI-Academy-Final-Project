import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.sanity_checks import run_sanity_checks
from src.experiments import (
    run_synthetic_experiment,
    run_digits_experiment,
    run_track_a,
)

DATA_DIR = ROOT_DIR / "data"


def main():
    t0 = time.time()

    run_sanity_checks()

    run_synthetic_experiment("Linear Gaussian", DATA_DIR / "linear_gaussian.npz")
    run_synthetic_experiment("Moons", DATA_DIR / "moons.npz")

    digits_results = run_digits_experiment()

    run_track_a(digits_results)

    print(f"\nAll done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()