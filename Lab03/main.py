import os
import pathlib
import time
from logic.dataset import retrieve_datasets
from logic.model import PredictionModel

ROOT_DIR_PATH = pathlib.Path(__file__).parent.resolve()
MODELS_DIR_NAME = "models"
MODELS_DIR_PATH = os.path.join(ROOT_DIR_PATH, MODELS_DIR_NAME)


if __name__ == "__main__":
    train_inputs, train_labels, test_inputs, test_labels = retrieve_datasets()

    model = PredictionModel()

    # Train and test a new model

    print("Testing accuracy before training...")
    model.test_accuracy(test_inputs, test_labels)

    start_time = time.time()
    print("Started training...")
    model.train(train_inputs, train_labels, 100, 100, 0.02)
    end_time = time.time()
    print(f"Training took {round(end_time - start_time)} seconds.")

    print("Testing accuracy after training...")
    model.test_accuracy(test_inputs, test_labels)

    model_serialization_path = os.path.join(MODELS_DIR_PATH, f"model_{int(time.time() * 1000)}")
    model.serialize(model_serialization_path)

    # Test an old model

    # Accuracy: 97.1 %
    # model.deserialize(r"D:\FII\NN\Lab03\models\model_1731007846379.npz")
    # model.test_accuracy(test_inputs, test_labels)
