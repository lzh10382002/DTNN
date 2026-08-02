"""Regression check against predictions saved by the original TF1 project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dtnn_web_model import DTNNModel


GOLDEN_CASES = {
    "rf_q_model_1pct": [
        ([40.0, 1.0, 4.0, 0.6], 1.0),
        ([40.0, 6.0, 30.0, 2.0], 0.5000656745),
        ([40.0, 7.0, 6.0, 0.6], 0.0),
        ([80.0, 7.0, 30.0, 0.6], 0.2500069453),
        ([120.0, 8.0, 10.0, 2.0], 0.4999998929),
        ([160.0, 8.0, 30.0, 1.6], 0.0999931335),
        ([240.0, 8.0, 30.0, 1.3], 1.0),
    ],
    "rf_q_model_2pct": [
        ([40.0, 1.0, 4.0, 0.602059991], 1.0),
        ([40.0, 4.0, 30.0, 1.0], 0.5001072727),
        ([40.0, 8.0, 4.0, 0.602059991], 0.0),
        ([80.0, 7.0, 10.0, 1.602059991], 0.5000001180),
        ([120.0, 8.0, 20.0, 1.602059991], 0.0999999395),
        ([120.0, 8.0, 20.0, 1.301029996], 0.2000096365),
        ([240.0, 8.0, 30.0, 1.301029996], 1.0),
    ],
}


def main() -> int:
    tolerance = 5e-7
    failures = []

    for model_name, cases in GOLDEN_CASES.items():
        model = DTNNModel(PROJECT_DIR / "models" / model_name)
        try:
            inputs = np.asarray([case[0] for case in cases], dtype=np.float64)
            expected = np.asarray([case[1] for case in cases], dtype=np.float64)
            actual = model.predict(inputs).reshape(-1)
            errors = np.abs(actual - expected)
            max_error = float(np.max(errors))
            print(f"{model_name}: max absolute error={max_error:.3e}")
            if max_error > tolerance:
                failures.append(f"{model_name}: {max_error:.3e} > {tolerance:.3e}")

            boundary_inputs = np.asarray(
                [[0.0, 1.0, 4.0, 0.6], [301.0, 8.0, 30.0, 2.0]],
                dtype=np.float64,
            )
            boundary_outputs = model.predict(boundary_inputs).reshape(-1)
            if not np.allclose(boundary_outputs, [0.0, 1.0], atol=0.0, rtol=0.0):
                failures.append(
                    f"{model_name}: depth constraints returned {boundary_outputs.tolist()}"
                )
        finally:
            model.close()

    if failures:
        print("Regression test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Regression test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

