import sys
import os

# Add parent directory to sys.path to allow imports from the main directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import EMGModel

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'emg_features_cc.csv'

    print(f"Loading data from {file_path}...")
    X, y = EMGModel.load_and_preprocess_data(file_path)

    if X is not None and y is not None:
        print(f"Data loaded. Shape: X={X.shape}, y={y.shape}")

        X_train, X_test, y_train, y_test = EMGModel.split_data(X, y)
        print(f"Data split. Train: {X_train.shape}, Test: {X_test.shape}")

        print("Building model...")
        emg = EMGModel(input_dim=X.shape[1])
        emg.build(verbose=True)

        print("Training model...")
        history = emg.train(X_train, y_train, epochs=50)

        print("Saving training graphs...")
        EMGModel.save_training_graphs(history)

        print("Evaluating model...")
        accuracy = emg.evaluate(X_test, y_test)
        print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

        emg.save()
