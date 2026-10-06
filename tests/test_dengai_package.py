"""The refactored ``dengai`` package must reproduce the legacy pipelines."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dengai import config, features, models
from dengai.data import city_frame, interpolate_climate, load_data, make_submission
from dengai.validation import interpolated_cities, temporal_folds

DATA_DIR = ROOT / "data"
LEGACY = {
    features.RAW_LAGS: "raw_lags",
    features.MULTISCALE: "multiscale_summaries",
    features.RAW_PLUS_MULTISCALE: "raw_plus_summaries",
}


class PackageMatchesLegacy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test, cls.template = load_data(DATA_DIR)
        cls.cities = interpolated_cities(cls.train)
        cls.test_cities = {c: interpolate_climate(city_frame(cls.test, c)) for c in config.CITIES}
        cls.stats = features.normalization_stats(cls.cities["sj"])

    def test_history_matrices_match_feature_lag_mlp(self):
        from feature_lag_mlp.representations import (
            build_representation_test_matrix,
            build_representation_training_matrix,
        )

        for city in config.CITIES:
            for representation, legacy_name in LEGACY.items():
                with self.subTest(city=city, representation=representation):
                    x, y = features.training_matrix(self.cities[city], city, self.stats, representation)
                    legacy_x, legacy_y = build_representation_training_matrix(self.train, city, legacy_name)
                    np.testing.assert_allclose(x, legacy_x, atol=1e-5)
                    np.testing.assert_array_equal(y, legacy_y)
                    test_x = features.prediction_matrix(
                        self.cities[city], self.test_cities[city], city, self.stats, representation
                    )
                    legacy_test = build_representation_test_matrix(self.train, self.test, city, legacy_name)
                    np.testing.assert_allclose(test_x, legacy_test, atol=1e-5)
                    self.assertEqual(x.shape[1], len(features.feature_names(city, representation)))

    def test_input_dimensions(self):
        self.assertEqual(len(features.raw_lag_names("sj")), 461)
        self.assertEqual(len(features.raw_lag_names("iq")), 380)
        self.assertEqual(len(features.multiscale_names()), 180)

    def test_tree_features_match_tree_ensemble(self):
        import pandas as pd
        from train_tree_ensemble import make_features

        for city in config.CITIES:
            frame = city_frame(self.train, city)
            pd.testing.assert_frame_equal(features.tree_features(frame), make_features(frame))

    def test_tree_features_are_past_only(self):
        frame = city_frame(self.train, "sj")
        base = features.tree_features(frame)
        changed = frame.copy()
        changed.loc[500:, "station_precip_mm"] += 1000.0
        after = features.tree_features(changed)
        lag_columns = [c for c in base if c.startswith("station_precip_mm__")]
        np.testing.assert_array_equal(base.loc[:500, lag_columns], after.loc[:500, lag_columns])

    def test_numpy_mlp_matches_research_implementation(self):
        sys.path.insert(0, str(ROOT))
        from feature_lag_mlp.config import ModelConfig, TrainingSchedule
        from research.numpy_mlp import NumpyFeatureLagMlp

        x, y = features.training_matrix(self.cities["iq"], "iq", self.stats, features.MULTISCALE)
        schedule = config.TrainingSchedule(epochs=3, steps_per_epoch=20)
        cfg = config.FINAL_MLP_CONFIGS["iq"]
        ours = models.SeluMLP("iq", cfg, schedule, seed=7).fit(x, y)
        legacy = NumpyFeatureLagMlp(
            "iq", ModelConfig(**cfg.to_dict()), TrainingSchedule(epochs=3, steps_per_epoch=20), 7
        ).fit(x, y)
        np.testing.assert_allclose(ours.predict(x[:50]), legacy.predict(x[:50]))

    def test_folds_are_chronological(self):
        for city in config.CITIES:
            for fold in temporal_folds(self.cities, city):
                self.assertLess(fold.train[config.DATE_COLUMN].max(), fold.valid[config.DATE_COLUMN].min())
                self.assertLess(fold.sj_reference[config.DATE_COLUMN].max(), fold.valid[config.DATE_COLUMN].min())
                self.assertEqual(len(fold.valid), 52)

    def test_submission_matches_template(self):
        predictions = {c: np.zeros(int(self.template["city"].eq(c).sum()), int) for c in config.CITIES}
        submission = make_submission(self.template, predictions)
        self.assertTrue(submission[config.KEY_COLUMNS].equals(self.template[config.KEY_COLUMNS]))


if __name__ == "__main__":
    unittest.main()
