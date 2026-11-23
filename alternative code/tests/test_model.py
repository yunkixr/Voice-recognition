from src.model import predict, half_mse_loss, compute_gradients, fit_standardizer, standardize, predict_label
from src.gradient_descent import gradient_descent


# Contoh kecil yang bisa dihitung dengan tangan.
X = [[1, 2], [2, 0], [3, 1]]
y = [0, 1, 1]


def close(a, b):
    return abs(a - b) < 0.00001


def main():
    weights, bias, history = gradient_descent(X, y, learning_rate=0.1, max_epochs=1)
    assert close(history[0], 1 / 3)
    assert close(weights[0], 1 / 6)
    assert close(weights[1], 1 / 30)
    assert close(bias, 1 / 15)
    print("PASS: First gradient descent iteration")

    scores = predict(X, weights, bias)
    assert all(close(a, b) for a, b in zip(scores, [0.3, 0.4, 0.6]))
    assert close(half_mse_loss(scores, y), 0.61 / 6)
    print("PASS: Prediction and loss")

    dw, db = compute_gradients(X, y, [0, 0, 0])
    assert close(dw[0], -5 / 3)
    assert close(dw[1], -1 / 3)
    assert close(db, -2 / 3)
    print("PASS: Gradients")

    mean, std = fit_standardizer(X)
    scaled = standardize(X, mean, std)
    assert all(close(sum(row[j] for row in scaled), 0) for j in range(2))
    assert predict_label([0.2, 0.5, 0.9]) == [0, 1, 1]
    print("PASS: Standardization and speaker labels")

    _, _, losses = gradient_descent(X, y, learning_rate=0.1, max_epochs=200)
    assert losses[-1] < losses[0]
    print("PASS: Loss decreases")

    # Cek turunan dengan finite difference: [J(w+h) - J(w-h)] / (2h).
    weights = [0.2, -0.4]
    bias = 0.3
    h = 0.000001
    dw, db = compute_gradients(X, y, predict(X, weights, bias))

    for j in range(len(weights)):
        plus = weights[:]
        minus = weights[:]
        plus[j] += h
        minus[j] -= h
        loss_plus = half_mse_loss(predict(X, plus, bias), y)
        loss_minus = half_mse_loss(predict(X, minus, bias), y)
        numerical_gradient = (loss_plus - loss_minus) / (2 * h)
        assert close(dw[j], numerical_gradient)

    loss_plus = half_mse_loss(predict(X, weights, bias + h), y)
    loss_minus = half_mse_loss(predict(X, weights, bias - h), y)
    assert close(db, (loss_plus - loss_minus) / (2 * h))
    print("PASS: Gradient matches finite difference")

    # Kalau suatu feature konstan, standardisasinya tidak boleh membagi nol.
    constant_X = [[3, 1], [3, 2], [3, 3]]
    mean, std = fit_standardizer(constant_X)
    assert std[0] == 1.0
    assert all(row[0] == 0 for row in standardize(constant_X, mean, std))
    print("PASS: Constant feature")

    try:
        half_mse_loss([], [])
        assert False, "Empty labels should raise an error"
    except ValueError:
        pass
    print("PASS: Empty loss input")

    print("\n8 tests passed")


if __name__ == "__main__":
    main()
