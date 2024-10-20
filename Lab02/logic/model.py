import numpy as np
from logic.engine import softmax, gradient_descent


class PredictionModel:
    def __init__(self):
        self.weights = np.random.randn(784, 10) * 0.01
        self.biases = np.zeros((10,))

    def predict(self, inputs: np.array) -> np.array:
        return softmax(np.dot(inputs, self.weights) + self.biases)

    def train(self, inputs: np.array, labels: np.array, epochs: int, batch_size: int, learning_rate: float) -> None:
        print("Training started!")

        samples_size = inputs.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(samples_size)

            inputs_shuffled = inputs[indices]
            labels_shuffled = labels[indices]

            for i in range(0, samples_size, batch_size):
                inputs_batch = inputs_shuffled[i: i + batch_size]
                labels_batch = labels_shuffled[i: i + batch_size]

                predictions_batch = self.predict(inputs_batch)

                weights_updates_batch, biases_updates_batch = gradient_descent(inputs_batch, labels_batch,
                                                                               predictions_batch)

                self.weights += learning_rate * weights_updates_batch
                self.biases += learning_rate * biases_updates_batch

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}")

        print("Training finished!")

    def test_accuracy(self, inputs: np.array, labels: np.array) -> np.floating:
        print("Accuracy test started!")

        predictions = self.predict(inputs)

        predicted_classes = np.argmax(predictions, axis=1)
        label_classes = np.argmax(labels, axis=1)

        accuracy = np.mean(predicted_classes == label_classes)

        print("Accuracy test finished!")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    def serialize(self, path: str) -> None:
        np.savez(path, weights=self.weights, biases=self.biases)

    def deserialize(self, path: str) -> None:
        data = np.load(path)

        self.weights = data["weights"]
        self.biases = data["biases"]
