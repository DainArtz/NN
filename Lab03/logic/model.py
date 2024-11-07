import numpy as np
import time
from logic.engine import sigmoid, softmax, sigmoid_derivative


class PredictionModel:
    def __init__(self):
        self.weights_hidden_layer = np.random.randn(784, 100)
        self.biases_hidden_layer = np.random.randn(100)

        self.weights_output_layer = np.random.randn(100, 10)
        self.biases_output_layer = np.random.randn(10)

    def predict(self, inputs: np.array) -> np.array:
        outputs_hidden_layer = inputs.dot(self.weights_hidden_layer) + self.biases_hidden_layer
        activated_outputs_hidden_layer = sigmoid(outputs_hidden_layer)
        outputs_output_layer = activated_outputs_hidden_layer.dot(self.weights_output_layer) + self.biases_output_layer
        activated_outputs_output_layer = softmax(outputs_output_layer)

        return activated_outputs_output_layer

    def perform_backpropagation(self, inputs: np.array, labels: np.array, learning_rate: float) -> None:
        active_hidden_layer_nodes = np.random.choice([0, 1], size=(self.weights_hidden_layer.shape[1],),
                                                     p=[1 - 0.8, 0.8])
        active_hidden_layer_nodes = np.tile(active_hidden_layer_nodes, (inputs.shape[0], 1))

        outputs_hidden_layer = np.dot(inputs, self.weights_hidden_layer) + self.biases_hidden_layer
        activated_outputs_hidden_layer = sigmoid(outputs_hidden_layer)
        activated_outputs_hidden_layer *= active_hidden_layer_nodes

        outputs_output_layer = (np.dot(activated_outputs_hidden_layer, self.weights_output_layer) +
                                self.biases_output_layer)
        activated_outputs_output_layer = softmax(outputs_output_layer)

        delta_output_layer = labels - activated_outputs_output_layer
        gradient_weights_output_layer = np.dot(activated_outputs_hidden_layer.T, delta_output_layer)
        gradient_biases_output_layer = np.sum(delta_output_layer, axis=0)

        delta_hidden_layer = (np.dot(delta_output_layer, self.weights_output_layer.T) *
                              sigmoid_derivative(outputs_hidden_layer))
        delta_hidden_layer *= active_hidden_layer_nodes
        gradient_weights_hidden_layer = np.dot(inputs.T, delta_hidden_layer)
        gradient_biases_hidden_layer = np.sum(delta_hidden_layer, axis=0)

        self.weights_hidden_layer += gradient_weights_hidden_layer * learning_rate
        self.biases_hidden_layer += gradient_biases_hidden_layer * learning_rate
        self.weights_output_layer += gradient_weights_output_layer * learning_rate
        self.biases_output_layer += gradient_biases_output_layer * learning_rate

    def train(self, inputs: np.array, labels: np.array, epochs: int, batch_size: int, learning_rate: float) -> None:
        for epoch in range(epochs):
            time_start = time.time()

            permutation = np.random.permutation(inputs.shape[0])
            shuffled_inputs = inputs[permutation]
            shuffled_labels = labels[permutation]

            for i in range(0, inputs.shape[0], batch_size):
                batch_inputs = shuffled_inputs[i:i + batch_size]
                batch_labels = shuffled_labels[i:i + batch_size]
                self.perform_backpropagation(batch_inputs, batch_labels, learning_rate)

            time_end = time.time()

            print(f"Finished epoch: #{epoch + 1} / #{epochs} - Elapsed time: {round(time_end-time_start, 2)} sec")

    def test_accuracy(self, inputs: np.array, labels: np.array, toggle_logging: bool = True) -> np.floating:
        predictions = self.predict(inputs)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)

        if toggle_logging:
            print(f"Accuracy: {accuracy * 100}%")

        return accuracy

    def serialize(self, path: str) -> None:
        np.savez(path,
                 weights_hidden_layer=self.weights_hidden_layer,
                 biases_hidden_layer=self.biases_hidden_layer,
                 weights_output_layer=self.weights_output_layer,
                 biases_output_layer=self.biases_output_layer)

    def deserialize(self, path: str) -> None:
        data = np.load(path)

        self.weights_hidden_layer = data["weights_hidden_layer"]
        self.biases_hidden_layer = data["biases_hidden_layer"]
        self.weights_output_layer = data["weights_output_layer"]
        self.biases_output_layer = data["biases_output_layer"]
