import numpy as np


def predict(X, weights, bias):
    return X @ weights + bias


def half_mse_loss(predictions, y):
    errors = predictions - y
    return np.mean(errors ** 2) / 2


def compute_gradients(X, y, predictions):
    errors = predictions - y
    dw = X.T @ errors / len(y)
    db = np.mean(errors)
    return dw, db


def fit_standardizer(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize(X, mean, std):
    return (X - mean) / std


def predict_label(scores, threshold=0.5):
    return (scores >= threshold).astype(int)
