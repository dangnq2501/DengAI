from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from research.baselines import seasonal_median_predict
from research.rounding import blend_raw, to_submission_cases


class ResearchHelperTests(unittest.TestCase):
    def test_seasonal_median_uses_local_week_window(self) -> None:
        weeks = np.array([1, 2, 3, 4, 5], dtype=float)
        cases = np.array([10, 20, 30, 40, 50], dtype=float)
        prediction = seasonal_median_predict(weeks, cases, np.array([3.0]))
        self.assertEqual(prediction.shape, (1,))
        self.assertEqual(float(prediction[0]), 30.0)

    def test_rounding_modes(self) -> None:
        raw = np.array([0.4, 0.5, 1.6, 2.5])
        np.testing.assert_array_equal(
            to_submission_cases(raw, mode="nearest"),
            np.array([0, 1, 2, 3]),
        )
        np.testing.assert_array_equal(
            to_submission_cases(raw, mode="floor"),
            np.array([0, 0, 1, 2]),
        )
        np.testing.assert_array_equal(
            to_submission_cases(raw, mode="truncate"),
            np.array([0, 0, 1, 2]),
        )

    def test_blend_raw(self) -> None:
        primary = np.array([10.0, 20.0])
        secondary = np.array([0.0, 40.0])
        blended = blend_raw(primary, secondary, 0.75)
        np.testing.assert_allclose(blended, np.array([7.5, 25.0]))


if __name__ == "__main__":
    unittest.main()
