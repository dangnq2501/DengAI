from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

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
from train_sj_garchx_ensemble import EgarchX, arch_lm_test
from train_boosting_ensemble import boosting_model
from train_feature_engineering import FEATURE_MODES, make_feature_mode
from train_sj_recency_ensemble import (
    HORIZON,
    recency_training_indices_and_weights,
    rolling_origins,
)
from train_sj_quantile_forest import per_tree_predictions
from nested_validate_sj import all_origins, inner_origins
from nested_validate_sj_star import _restore_natural_prior
from train_vendor_history_nn import MatrixBuilder, SeluMAENetwork
from reproduce_vendor_17_57 import EXPECTED_INPUTS, WINDOWS
from train_vendor_v10_reproduction import build_training_matrix


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

    def test_vendor_history_dimensions_and_shift(self) -> None:
        city = (
            self.train.loc[self.train["city"].eq("sj")]
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
            .iloc[:180]
            .copy()
        )
        builder = MatrixBuilder(city, "sj", 128, enhanced=False)
        x, y = builder.training(city.iloc[:128][TARGET_COLUMN].to_numpy(float))
        self.assertEqual(x.shape, (128, 110))
        self.assertEqual(y.shape, (128,))
        shifted = builder.prediction(128, 52, shift=2)
        aligned = builder.prediction(128, 52, shift=0)
        self.assertEqual(shifted.shape, aligned.shape)
        np.testing.assert_allclose(shifted[2], aligned[0])

    def test_numpy_selu_network_is_finite(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(48, 8))
        y = np.maximum(3.0 + x[:, 0] - x[:, 1], 0.0)
        model = SeluMAENetwork(
            hidden_units=8,
            epochs=2,
            steps_per_epoch=4,
            random_state=42,
        ).fit(x, y)
        prediction = model.predict(x[:5])
        self.assertEqual(prediction.shape, (5,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_saved_vendor_model_input_dimensions(self) -> None:
        for city in ("iq", "sj"):
            self.assertEqual(sum(WINDOWS[city].values()), EXPECTED_INPUTS[city])

    def test_vendor_v10_training_matrix_dimensions(self) -> None:
        for city, rows in (("sj", 936), ("iq", 520)):
            matrix, target = build_training_matrix(self.train, city)
            self.assertEqual(matrix.shape, (rows, EXPECTED_INPUTS[city]))
            self.assertEqual(target.shape, (rows,))

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

    def test_egarchx_produces_positive_forecast_volatility(self) -> None:
        rng = np.random.default_rng(42)
        residual = rng.normal(size=80)
        exogenous = pd.DataFrame(
            {"temperature": rng.normal(size=80), "rain": rng.normal(size=80)}
        )
        model = EgarchX(max_iter=50).fit(residual, exogenous)
        sigma = model.forecast_sigma(exogenous.iloc[:12])
        self.assertEqual(sigma.shape, (12,))
        self.assertTrue(np.isfinite(sigma).all())
        self.assertTrue((sigma > 0).all())
        diagnostic = arch_lm_test(residual, lags=4)
        self.assertTrue(0.0 <= diagnostic["p_value"] <= 1.0)

    def test_boosting_factories(self) -> None:
        xgb = boosting_model("xgb_poisson_d2", iterations=2, random_state=42)
        cat = boosting_model("cat_mae_d4", iterations=2, random_state=42)
        self.assertIsInstance(xgb, XGBRegressor)
        self.assertIsInstance(cat, CatBoostRegressor)

    def test_engineered_features_are_past_only(self) -> None:
        city = (
            self.train.loc[self.train["city"].eq("sj")]
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
            .iloc[:180]
            .copy()
        )
        original = make_feature_mode(city, "all")
        changed = city.copy()
        changed.loc[140:, "precipitation_amt_mm"] = 1_000_000.0
        modified = make_feature_mode(changed, "all")
        # Raw current weather may change from row 140 onward, but no earlier row
        # may depend on those future values.
        np.testing.assert_allclose(
            original.iloc[:140].to_numpy(),
            modified.iloc[:140].to_numpy(),
            equal_nan=True,
        )
        self.assertTrue(set(FEATURE_MODES) >= {"baseline", "biology", "all"})

    def test_sj_recency_splits_match_competition_horizon(self) -> None:
        origins = rolling_origins(936)
        self.assertEqual(origins, [416, 468, 520, 572, 624, 676])
        self.assertEqual(936 - origins[-1], HORIZON)
        indices, weights = recency_training_indices_and_weights(
            origins[-1], {"window_weeks": None, "half_life_years": 5.0}
        )
        self.assertEqual(len(indices), origins[-1])
        self.assertEqual(len(weights), origins[-1])
        self.assertAlmostEqual(float(weights[-1]), 1.0)
        self.assertLess(float(weights[0]), float(weights[-1]))

    def test_sj_per_tree_predictions(self) -> None:
        city = (
            self.train.loc[self.train["city"].eq("sj")]
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
            .iloc[:120]
        )
        features = make_features(city)
        model = tree_model("sj", n_estimators=3)
        model.fit(features.iloc[:100], city[TARGET_COLUMN].iloc[:100])
        prediction = per_tree_predictions(model, features.iloc[100:])
        self.assertEqual(prediction.shape, (3, 20))
        self.assertTrue(np.isfinite(prediction).all())

    def test_nested_sj_inner_blocks_end_before_outer_test(self) -> None:
        available = all_origins(936)
        self.assertEqual(available, [260, 312, 364, 416, 468, 520, 572, 624, 676])
        for outer in (520, 572, 624, 676):
            inner = inner_origins(outer, available)
            self.assertTrue(inner)
            self.assertTrue(all(origin + 260 <= outer for origin in inner))
            self.assertNotIn(outer, inner)

    def test_star_gate_prior_restoration(self) -> None:
        balanced = np.array([0.2, 0.5, 0.8])
        natural = _restore_natural_prior(balanced, prevalence=0.10)
        self.assertTrue((natural > 0).all() and (natural < 1).all())
        self.assertTrue((natural < balanced).all())


if __name__ == "__main__":
    unittest.main()
