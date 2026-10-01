from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from train_tree_ensemble import (
    DATE_COLUMN,
    DEFAULT_MODEL_PARAMS,
    KEY_COLUMNS,
    TARGET_COLUMN,
    _grid_configs,
    load_data,
    make_features,
    tree_model,
)
from train_sj_nb_ensemble import NegativeBinomialRegressor, select_blend


class TreePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train, cls.test, cls.template = load_data(PROJECT_DIR / "data")

    def test_input_data_and_template_align(self) -> None:
        self.assertEqual(len(self.train), 1456)
        self.assertEqual(len(self.test), 416)
        pd.testing.assert_frame_equal(
            self.test[KEY_COLUMNS].reset_index(drop=True),
            self.template[KEY_COLUMNS].reset_index(drop=True),
        )
        self.assertTrue((self.train[TARGET_COLUMN] >= 0).all())

    def test_features_exclude_identifiers_and_target(self) -> None:
        city = self.train.loc[self.train["city"].eq("iq")].reset_index(drop=True)
        features = make_features(city)
        forbidden = {"city", DATE_COLUMN, TARGET_COLUMN}
        self.assertFalse(forbidden & set(features))
        self.assertIn("year", features)
        self.assertIn("weekofyear", features)
        self.assertIn("week_sin_4", features)
        self.assertIn("station_avg_temp_c__lag_16", features)

    def test_lagged_features_do_not_use_future_rows(self) -> None:
        city = (
            self.train.loc[self.train["city"].eq("sj")]
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
            .iloc[:80]
            .copy()
        )
        original = make_features(city)
        changed = city.copy()
        changed.loc[60:, "station_avg_temp_c"] = 1_000_000.0
        modified = make_features(changed)
        np.testing.assert_allclose(
            original.iloc[:60].to_numpy(),
            modified.iloc[:60].to_numpy(),
            equal_nan=True,
        )

    def test_city_models_are_the_confirmed_tree_estimators(self) -> None:
        iq = repr(tree_model("iq", n_estimators=2))
        sj = repr(tree_model("sj", n_estimators=2))
        self.assertIn("RandomForestRegressor", iq)
        self.assertIn("TransformedTargetRegressor", iq)
        self.assertIn("ExtraTreesRegressor", sj)
        self.assertIn("criterion='poisson'", sj)
        self.assertNotIn("GradientBoosting", iq + sj)

    def test_tuning_grid_contains_defaults_and_model_accepts_parameters(self) -> None:
        for city in ("iq", "sj"):
            self.assertIn(DEFAULT_MODEL_PARAMS[city], _grid_configs(city))
        iq = repr(
            tree_model(
                "iq",
                n_estimators=2,
                model_params={
                    "max_features": 0.6,
                    "min_samples_leaf": 7,
                    "max_depth": 20,
                },
            )
        )
        self.assertIn("max_depth=20", iq)
        self.assertIn("max_features=0.6", iq)
        self.assertIn("min_samples_leaf=7", iq)

    def test_confirmed_submission_is_valid(self) -> None:
        submission = pd.read_csv(
            PROJECT_DIR / "artifacts" / "submission_tree_ensemble.csv"
        )
        self.assertEqual(submission.columns.tolist(), KEY_COLUMNS + [TARGET_COLUMN])
        pd.testing.assert_frame_equal(
            submission[KEY_COLUMNS], self.template[KEY_COLUMNS]
        )
        self.assertTrue(submission[TARGET_COLUMN].notna().all())
        self.assertTrue((submission[TARGET_COLUMN] >= 0).all())
        self.assertEqual(
            submission.groupby("city")[TARGET_COLUMN].max().to_dict(),
            {"iq": 16, "sj": 99},
        )

    def test_negative_binomial_regressor_produces_nonnegative_means(self) -> None:
        features = pd.DataFrame(
            {"year": np.arange(30), "signal": np.linspace(-1.0, 1.0, 30)}
        )
        target = np.arange(30) % 7
        model = NegativeBinomialRegressor(max_iter=100).fit(features, target)
        prediction = model.predict(features)
        self.assertEqual(prediction.shape, target.shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())


if __name__ == "__main__":
    unittest.main()
