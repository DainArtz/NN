import numpy as np


class PredictionModel:
    def __init__(self):
        pass

    def predict(self, inputs: np.array) -> np.array:
        pass

    def train(self, inputs: np.array, labels: np.array, epochs: int, batch_size: int, learning_rate: float) -> None:
        pass

    def test_accuracy(self, inputs: np.array, labels: np.array, toggle_logging: bool = True) -> np.floating:
        pass

    def serialize(self, path: str) -> None:
        pass

    def deserialize(self, path: str) -> None:
        pass
