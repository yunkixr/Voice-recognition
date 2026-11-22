import unittest

import numpy as np

from src.model import (
    predict,
    half_mse_loss,
    compute_gradients,
    fit_standardizer,
    standardize,
    predict_label
)
from src.gradient_descent import gradient_descent


# Tiny dataset from the hand calculation in docs/regina_math.md
X_TINY = np.array([
    [1.0, 2.0],
    [2.0, 0.0],
    [3.0, 1.0]
])
Y_TINY = np.array([0.0, 1.0, 1.0])


class TestOneIterationByHand(unittest.TestCase):

    def test_first_iteration(self):
        weights, bias, loss_history = gradient_descent(
            X_TINY, Y_TINY, learning_rate=0.1, max_epochs=1
        )

        # Start at w = [0, 0], b = 0, so every prediction is 0.
        self.assertAlmostEqual(loss_history[0], 1 / 3)

        # dw = [-5/3, -1/3], db = -2/3, then w = 0 - 0.1 * dw, b = 0 - 0.1 * db
        np.testing.assert_allclose(weights, [1 / 6, 1 / 30])
        self.assertAlmostEqual(bias, 1 / 15)

    def test_loss_after_first_update(self):
        weights = np.array([1 / 6, 1 / 30])
        bias = 1 / 15

        y_pred = predict(X_TINY, weights, bias)
        np.testing.assert_allclose(y_pred, [0.3, 0.4, 0.6])
        self.assertAlmostEqual(half_mse_loss(y_pred, Y_TINY), 0.61 / 6)


class TestGradients(unittest.TestCase):

    def test_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 3))
        y = rng.integers(0, 2, size=8).astype(float)
        weights = rng.normal(size=3)
        bias = 0.3
        h = 1e-6

        dw, db = compute_gradients(X, y, predict(X, weights, bias))

        # Central difference: (J(w + h) - J(w - h)) / 2h for each weight
        for j in range(3):
            w_plus = weights.copy()
            w_minus = weights.copy()
            w_plus[j] += h
            w_minus[j] -= h
            loss_plus = half_mse_loss(predict(X, w_plus, bias), y)
            loss_minus = half_mse_loss(predict(X, w_minus, bias), y)
            self.assertAlmostEqual(dw[j], (loss_plus - loss_minus) / (2 * h), places=6)

        loss_plus = half_mse_loss(predict(X, weights, bias + h), y)
        loss_minus = half_mse_loss(predict(X, weights, bias - h), y)
        self.assertAlmostEqual(db, (loss_plus - loss_minus) / (2 * h), places=6)


class TestConvergence(unittest.TestCase):

    def test_loss_never_increases_with_small_learning_rate(self):
        _, _, loss_history = gradient_descent(
            X_TINY, Y_TINY, learning_rate=0.1, max_epochs=200, tolerance=0
        )
        for before, after in zip(loss_history, loss_history[1:]):
            self.assertLessEqual(after, before)

    def test_matches_least_squares_solution(self):
        weights, bias, _ = gradient_descent(
            X_TINY, Y_TINY, learning_rate=0.1, max_epochs=20000, tolerance=1e-15
        )

        # lstsq is only used to check the answer; it solves [X 1] @ [w b] = y exactly.
        X_with_ones = np.column_stack([X_TINY, np.ones(len(X_TINY))])
        exact = np.linalg.lstsq(X_with_ones, Y_TINY, rcond=None)[0]

        np.testing.assert_allclose(weights, exact[:2], atol=1e-5)
        self.assertAlmostEqual(bias, exact[2], places=5)


class TestHelpers(unittest.TestCase):

    def test_standardize_gives_mean_zero_std_one(self):
        mean, std = fit_standardizer(X_TINY)
        X_scaled = standardize(X_TINY, mean, std)
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(X_scaled.std(axis=0), [1, 1])

    def test_threshold(self):
        labels = predict_label(np.array([-0.2, 0.49, 0.5, 1.3]))
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
