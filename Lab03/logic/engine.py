import numpy as np


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))


def sigmoid_derivative(x: np.array) -> np.array:
    value = sigmoid(x)
    return value * (1 - value)


def softmax(x: np.array) -> np.array:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
