import numpy as np
from torchvision.datasets import MNIST


def _encode_one_hot(labels: np.array) -> np.array:
    one_hot = np.zeros((labels.shape[0], 10))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


def _retrieve_dataset(is_train: bool) -> tuple[np.array, np.array]:
    dataset = MNIST(root="./data",
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

    data, labels = zip(*dataset)

    data = np.array(data) / 255.0
    labels = _encode_one_hot(np.array(labels))

    return data, labels


def retrieve_datasets() -> tuple[np.array, np.array, np.array, np.array]:
    (train_data, train_labels), (test_data, test_labels) = _retrieve_dataset(True), _retrieve_dataset(False)

    return train_data, train_labels, test_data, test_labels


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = retrieve_datasets()
    print(test_data[:20])
    print(test_labels[:20])
