import unittest
import numpy as np

from src.model import (predict, half_mse_loss, compute_gradients,
                       fit_standardizer, standardize, predict_label)
from src.gradient_descent import gradient_descent


X = np.array([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0]])
y = np.array([0.0, 1.0, 1.0])


class TestModel(unittest.TestCase):
    def test_one_iteration(self):
        weights, bias, history = gradient_descent(X, y, learning_rate=0.1, max_epochs=1)
        self.assertAlmostEqual(history[0], 1 / 3)
        np.testing.assert_allclose(weights, [1 / 6, 1 / 30])
        self.assertAlmostEqual(bias, 1 / 15)

    def test_predictions_and_loss(self):
        scores = predict(X, np.array([1 / 6, 1 / 30]), 1 / 15)
        np.testing.assert_allclose(scores, [0.3, 0.4, 0.6])
        self.assertAlmostEqual(half_mse_loss(scores, y), 0.61 / 6)

    def test_gradients(self):
        predictions = predict(X, np.zeros(2), 0)
        dw, db = compute_gradients(X, y, predictions)
        np.testing.assert_allclose(dw, [-5 / 3, -1 / 3])
        self.assertAlmostEqual(db, -2 / 3)

    def test_finite_difference(self):
        weights = np.array([0.2, -0.1])
        bias = 0.3
        dw, _ = compute_gradients(X, y, predict(X, weights, bias))
        h = 1e-6
        for j in range(2):
            plus, minus = weights.copy(), weights.copy()
            plus[j] += h
            minus[j] -= h
            numerical = (half_mse_loss(predict(X, plus, bias), y)
                         - half_mse_loss(predict(X, minus, bias), y)) / (2 * h)
            self.assertAlmostEqual(dw[j], numerical, places=6)

    def test_standardization(self):
        mean, std = fit_standardizer(X)
        scaled = standardize(X, mean, std)
        np.testing.assert_allclose(scaled.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(scaled.std(axis=0), [1, 1])

    def test_constant_feature(self):
        constant = np.array([[2., 3.], [2., 5.], [2., 7.]])
        mean, std = fit_standardizer(constant)
        self.assertEqual(std[0], 1.0)
        self.assertTrue(np.isfinite(standardize(constant, mean, std)).all())

    def test_convergence(self):
        _, _, history = gradient_descent(X, y, learning_rate=0.1,
                                         max_epochs=200, tolerance=0)
        self.assertTrue(all(b <= a for a, b in zip(history, history[1:])))

    def test_threshold(self):
        np.testing.assert_array_equal(predict_label(np.array([0.49, 0.5, 1.2])),
                                      [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
