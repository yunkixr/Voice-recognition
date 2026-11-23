def predict(X, weights, bias):
    results = []
    for row in X:
        score = bias
        for j in range(len(weights)):
            score += weights[j] * row[j]
        results.append(score)
    return results


def half_mse_loss(predictions, y):
    if not y or len(predictions) != len(y):
        raise ValueError("Predictions and labels must have the same nonzero length")

    total = 0.0
    for i in range(len(y)):
        error = predictions[i] - y[i]
        total += error ** 2
    return total / (2 * len(y))


def compute_gradients(X, y, predictions):
    n = len(y)
    dw = [0.0] * len(X[0])
    db = 0.0

    for i in range(n):
        error = predictions[i] - y[i]
        db += error
        for j in range(len(dw)):
            dw[j] += error * X[i][j]

    for j in range(len(dw)):
        dw[j] /= n
    return dw, db / n


def fit_standardizer(X):
    if not X:
        raise ValueError("Training data is empty")

    n = len(X)
    mean = []
    std = []

    for j in range(len(X[0])):
        avg = sum(row[j] for row in X) / n
        variance = sum((row[j] - avg) ** 2 for row in X) / n
        standard_deviation = variance ** 0.5

        if standard_deviation == 0:
            standard_deviation = 1.0

        mean.append(avg)
        std.append(standard_deviation)

    return mean, std


def standardize(X, mean, std):
    result = []
    for row in X:
        result.append([(row[j] - mean[j]) / std[j]
                       for j in range(len(row))])
    return result


def predict_label(scores, threshold=0.5):
    return [1 if score >= threshold else 0 for score in scores]
