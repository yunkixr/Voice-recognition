import numpy as np

from src.config import LEARNING_RATE, MAX_EPOCHS, TOLERANCE
from src.model import predict, half_mse_loss, compute_gradients


def gradient_descent(X, y, learning_rate=LEARNING_RATE,
                     max_epochs=MAX_EPOCHS, tolerance=TOLERANCE):
    weights = np.zeros(X.shape[1])
    bias = 0.0
    loss_history = []

    for epoch in range(max_epochs):
        y_pred = predict(X, weights, bias)
        loss = half_mse_loss(y_pred, y)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, y_pred)

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Converged: the loss barely changes between two iterations.
        if epoch > 0 and abs(loss_history[-2] - loss) < tolerance:
            break

        # Diverged: the learning rate is too large and the loss blew up.
        if not np.isfinite(loss) or loss > 1e10:
            break

    return weights, bias, loss_history
