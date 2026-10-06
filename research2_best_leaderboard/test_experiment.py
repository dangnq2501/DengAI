"""Check temporal isolation and submission ensemble semantics without training."""
import unittest
import numpy as np
from run_experiment import (aggregate, integer_cases, split_data, load_competition_data,
                            prepared_folds, ROOT, build_test_matrix)

class ExperimentTests(unittest.TestCase):
    def test_development_is_before_audit(self):
        train, test, _ = load_competition_data(ROOT / 'data')
        dev, audit = split_data(train, test)
        for city in ('sj', 'iq'):
            d = dev.loc[dev.city.eq(city)]
            a = audit.loc[audit.city.eq(city)]
            self.assertLess(d.week_start_date.max(), a.week_start_date.min())
            self.assertEqual(len(a), int(test.city.eq(city).sum()))
            for fold in prepared_folds(dev, city, 3, 52):
                self.assertEqual(len(fold['validation_y']), 52)
                self.assertTrue(np.isfinite(fold['train_x']).all())
                self.assertLess(fold['validation_end'], str(a.week_start_date.min().date()))
            x = build_test_matrix(dev, audit.drop(columns='total_cases'), city)
            mutated = audit.copy()
            mutated['total_cases'] = 100000
            np.testing.assert_array_equal(x, build_test_matrix(dev, mutated, city))

    def test_training_is_reproducible(self):
        from torch_backend import fit
        from feature_lag_mlp.config import PROFILE_CONFIGS
        rng = np.random.default_rng(10)
        x = rng.normal(size=(33, 8)).astype('float32')
        y = np.arange(33, dtype='float32')
        args = (x, y, x[:5], 'sj', PROFILE_CONFIGS['improved']['sj'], 42, 2, 2, [1, 2])
        first, history, _ = fit(*args)
        second, _, _ = fit(*args)
        self.assertEqual(first.shape, (2, 5))
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_aggregation_precedes_truncation(self):
        bank = np.ones((2, 3, 3, 5)) * 1.9
        bank[0, 0, -1] = 4.9
        predictions = aggregate(bank)
        np.testing.assert_allclose(predictions['control'], 4.9)
        np.testing.assert_array_equal(integer_cases(predictions['improved_mean']), [2]*5)
        np.testing.assert_array_equal(integer_cases([-5, .9, 2.9]), [0, 0, 2])
        with self.assertRaises(ValueError):
            integer_cases([np.nan])

if __name__ == '__main__':
    unittest.main()
