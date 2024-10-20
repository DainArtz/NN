import numpy as np


def softmax(weighted_sums: np.array) -> np.array:
    exponential_weighted_sums = np.exp(weighted_sums)
    return exponential_weighted_sums / np.sum(exponential_weighted_sums, axis=1, keepdims=True)


def cross_entropy(predictions: np.array, labels: np.array) -> np.array:
    epsilon = 1e-8
    batch_size = labels.shape[0]
    return -np.sum(labels * np.log(predictions + epsilon)) / batch_size


def gradient_descent(inputs: np.array, labels: np.array, predictions: np.array) -> tuple[np.array, np.array]:
    batch_size = labels.shape[0]
    loss_gradients = (labels - predictions) / batch_size
    weights_updates = np.dot(inputs.T, loss_gradients)
    biases_updates = np.sum(loss_gradients)

    return weights_updates, biases_updates
