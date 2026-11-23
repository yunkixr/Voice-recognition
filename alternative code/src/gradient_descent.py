from src.config import LEARNING_RATE, MAX_EPOCHS, TOLERANCE
from src.model import predict, half_mse_loss, compute_gradients


def gradient_descent(X, y, learning_rate=LEARNING_RATE,
                     max_epochs=MAX_EPOCHS, tolerance=TOLERANCE):
    weights = [0.0] * len(X[0])
    bias = 0.0
    history = []

    for epoch in range(max_epochs):
        predictions = predict(X, weights, bias)
        loss = half_mse_loss(predictions, y)
        history.append(loss)

        if loss > 1e10 or loss != loss:
            break
        if epoch > 0 and abs(history[-2] - loss) < tolerance:
            break

        dw, db = compute_gradients(X, y, predictions)
        for j in range(len(weights)):
            weights[j] -= learning_rate * dw[j]
        bias -= learning_rate * db

    return weights, bias, history
