import numpy as np

from src.config import LEARNING_RATE, MAX_EPOCHS, TOLERANCE
from src.model import predict, half_mse_loss, compute_gradients


def gradient_descent(X, y, learning_rate=LEARNING_RATE,
                     max_epochs=MAX_EPOCHS, tolerance=TOLERANCE):
    weights = np.zeros(X.shape[1])
    bias = 0.0
    loss_history = []

    for _ in range(max_epochs):
        predictions = predict(X, weights, bias)
        loss = half_mse_loss(predictions, y)

        if not np.isfinite(loss) or loss > 1e10:
            break

        loss_history.append(loss)
        if len(loss_history) > 1 and abs(loss_history[-2] - loss) < tolerance:
            break

        dw, db = compute_gradients(X, y, predictions)
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    return weights, bias, loss_history
