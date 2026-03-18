import sys
import os

# Add parent directory to sys.path to allow imports from the main directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from model import EMGModel

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'emg_features.csv'

    # --- Model / training parameters ---
    hidden_layers = 1
    neurons = 16
    dropout = 0.3708
    epochs = 50
    batch_size = 32
    test_size = 0.2
    random_state = 13
    loss_fn = "binary_crossentropy"
    optimizer = "adam"
    early_stop_monitor = "accuracy"
    early_stop_patience = 5

    # --- MLflow setup ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EMG")

    print(f"Loading data from {file_path}...")
    X, y = EMGModel.load_and_preprocess_data(file_path)

    if X is not None and y is not None:
        print(f"Data loaded. Shape: X={X.shape}, y={y.shape}")

        X_train, X_test, y_train, y_test = EMGModel.split_data(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Data split. Train: {X_train.shape}, Test: {X_test.shape}")

        with mlflow.start_run():
            # Log all parameters
            mlflow.log_params({
                # Data parameters
                "data_file": file_path,
                "input_dim": X.shape[1],
                "num_samples": X.shape[0],
                "test_size": test_size,
                "random_state": random_state,
                # Architecture parameters
                "hidden_layers": hidden_layers,
                "neurons": neurons,
                "dropout": dropout,
                "loss_function": loss_fn,
                "optimizer": optimizer,
                # Training parameters
                "epochs": epochs,
                "batch_size": batch_size,
                "early_stop_monitor": early_stop_monitor,
                "early_stop_patience": early_stop_patience,
                "validation_split": 0.2,
            })

            print("Building model...")
            emg = EMGModel(input_dim=X.shape[1])
            emg.build(
                hidden_layers=hidden_layers,
                neurons=neurons,
                dropout=dropout,
                verbose=True,
            )

            print("Training model...")
            history = emg.train(
                X_train, y_train, epochs=epochs, batch_size=batch_size
            )

            print("Evaluating model...")
            accuracy = emg.evaluate(X_test, y_test)
            print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

            # Log final metrics
            mlflow.log_metric("test_accuracy", accuracy)

            # Log per-epoch metrics from training history
            for epoch_idx in range(len(history.history['loss'])):
                mlflow.log_metrics({
                    "train_loss": history.history['loss'][epoch_idx],
                    "train_accuracy": history.history['accuracy'][epoch_idx],
                    "val_loss": history.history['val_loss'][epoch_idx],
                    "val_accuracy": history.history['val_accuracy'][epoch_idx],
                }, step=epoch_idx)

            emg.save()
            mlflow.log_artifact("bg_model.h5")

            print(f"MLflow run logged to experiment 'EMG'")
