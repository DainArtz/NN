import numpy as np


def sigmoid(inputs: np.array) -> np.array:
    return 1 / (1 + np.exp(-np.clip(inputs, -100, 100)))


def sigmoid_derivative(inputs: np.array) -> np.array:
    value = sigmoid(inputs)
    return value * (1 - value)


def softmax(inputs: np.array) -> np.array:
    exponential_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exponential_inputs / np.sum(exponential_inputs, axis=1, keepdims=True)
