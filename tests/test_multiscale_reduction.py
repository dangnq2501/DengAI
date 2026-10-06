from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from research.multiscale_reduction import ReductionSpec, reduce_multiscale


class MultiscaleReductionTests(unittest.TestCase):
    def test_block_pca_reduces_weather_blocks(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(size=(120, 180)).astype(np.float32)
        apply = rng.normal(size=(40, 180)).astype(np.float32)
        spec = ReductionSpec("block_pca_4", "block_pca", block_components=4)
        train_red, apply_red, state = reduce_multiscale(train, apply, spec)
        self.assertEqual(train_red.shape[0], 120)
        self.assertEqual(apply_red.shape[0], 40)
        self.assertEqual(train_red.shape[1], 16 * 4 + 4)
        self.assertIsNotNone(state)


if __name__ == "__main__":
    unittest.main()
