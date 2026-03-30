import time

from sanity_checks import run_sanity_checks
from experiments_softmax import (
    run_synthetic_experiment,
    run_digits_experiment,
    run_track_a,
)


def main():
    t0 = time.time()

    run_sanity_checks()

    run_synthetic_experiment("Linear Gaussian", "linear_gaussian")
    run_synthetic_experiment("Moons", "moons")

    digits_results = run_digits_experiment()

    run_track_a(digits_results)

    print(f"\nAll done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
