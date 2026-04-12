import sys
import os

# Add parent directory to sys.path to allow imports from the main directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
from config import INPUT_DIM


class EMGRandomForest:
    """Wraps a scikit-learn Random Forest Classifier for EMG binary classification.

    Mirrors the EMGModel API: build, train, evaluate, predict, save, load.
    """

    DEFAULT_INPUT_DIM = INPUT_DIM

    def __init__(self, model=None, input_dim=None):
        self.model = model
        self.input_dim = input_dim or self.DEFAULT_INPUT_DIM

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #
    def build(self, n_estimators=100, max_depth=None, min_samples_split=2, verbose=False):
        """Construct a Random Forest classifier.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        verbose : bool
            Print model details after building.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=13,
        )

        if verbose:
            print(f"Random Forest built: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")

        return self

    # ------------------------------------------------------------------ #
    #  Train
    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train):
        """Train the random forest model."""
        self.model.fit(X_train, y_train)
        train_acc = self.model.score(X_train, y_train)
        print(f"Training accuracy: {train_acc * 100:.2f}%")
        return self

    # ------------------------------------------------------------------ #
    #  Evaluate / Predict
    # ------------------------------------------------------------------ #
    def evaluate(self, X_test, y_test):
        """Return accuracy on the test set."""
        accuracy = self.model.score(X_test, y_test)
        return accuracy

    def predict(self, X):
        """Return a boolean prediction (True = swallow)."""
        return bool(self.model.predict(X)[0])

    def detailed_report(self, X_test, y_test):
        """Print classification report and confusion matrix."""
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Off", "Click"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def save(self, filepath='random_forest_model.pkl'):
        """Persist model to disk using joblib."""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load a saved model from *filepath* and return an instance."""
        if not os.path.exists(filepath):
            print(f"Error: Model '{filepath}' not found.")
            return None
        model = joblib.load(filepath)
        return cls(model=model)

    # ------------------------------------------------------------------ #
    #  Data helpers  (reuse EMGModel's static methods)
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_and_preprocess_data(filepath='emg_features.csv'):
        """Load a feature CSV and return (X, y) numpy arrays."""
        import pandas as pd
        from feature_engineering import FeatureEngineer

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None, None

        filt_aac = df['filt_AAC'].values.reshape(-1, 1)
        env_wl = df['env_WL'].values.reshape(-1, 1)
        filt_ar2 = df['filt_AR_2'].values.reshape(-1, 1)
        X = np.hstack([filt_aac, env_wl, filt_ar2])
        y = df['Output'].values
        return X, y

    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=13):
        """Split arrays into train/test sets."""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ====================================================================== #
#  Main – mirrors model_main.py
# ====================================================================== #
if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'emg_features.csv'

    # --- Model / training parameters ---
    n_estimators = 5000
    max_depth = 7
    min_samples_split = 2
    test_size = 0.2
    random_state = 13

    # --- MLflow setup ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EMG")

    print(f"Loading data from {file_path}...")
    X, y = EMGRandomForest.load_and_preprocess_data(file_path)

    if X is not None and y is not None:
        print(f"Data loaded. Shape: X={X.shape}, y={y.shape}")

        X_train, X_test, y_train, y_test = EMGRandomForest.split_data(
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
                # Model parameters
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            })

            print("Building model...")
            rf = EMGRandomForest(input_dim=X.shape[1])
            rf.build(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, verbose=True)

            print("Training model...")
            rf.train(X_train, y_train)

            print("Evaluating model...")
            accuracy = rf.evaluate(X_test, y_test)
            print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

            # Detailed report
            rf.detailed_report(X_test, y_test)

            # Log final metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("train_accuracy", rf.model.score(X_train, y_train))

            # Save model
            rf.save()
            mlflow.log_artifact("random_forest_model.pkl")

            print(f"MLflow run logged to experiment 'EMG'")
