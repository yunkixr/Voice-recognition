import numpy as np


def predict(X, weights, bias):
    # y_pred = w1*x1 + w2*x2 + w3*x3 + b, for every row of X at once
    return X @ weights + bias


def half_mse_loss(y_pred, y):
    N = len(y)
    errors = y_pred - y
    return np.sum(errors ** 2) / (2 * N)


def compute_gradients(X, y, y_pred):
    # dJ/dw = X^T (y_pred - y) / N    and    dJ/db = sum(y_pred - y) / N
    N = len(y)
    errors = y_pred - y
    dw = X.T @ errors / N
    db = np.sum(errors) / N
    return dw, db


def fit_standardizer(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return mean, std


def standardize(X, mean, std):
    return (X - mean) / std


def predict_label(scores, threshold=0.5):
    # Scores are not probabilities: >= 0.5 is closer to label 1 (B) than to 0 (A).
    return (scores >= threshold).astype(int)
