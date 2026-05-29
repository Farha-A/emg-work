import sys
import os

# Add parent directory to sys.path to allow imports from the main directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import datetime
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
from imblearn.over_sampling import BorderlineSMOTE
import mlflow
from config import INPUT_DIM


class EMGKNN:
    """Wraps a scikit-learn K-Nearest Neighbors Classifier for EMG binary classification.

    Mirrors the EMGModel API: build, train, evaluate, predict, save, load.
    """

    DEFAULT_INPUT_DIM = INPUT_DIM

    def __init__(self, model=None, input_dim=None):
        self.model = model
        self.input_dim = input_dim or self.DEFAULT_INPUT_DIM

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #
    def build(self, n_neighbors=5, weights='uniform', metric='minkowski', verbose=False):
        """Construct a K-Nearest Neighbors classifier.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to use.
        weights : str
            Weight function used in prediction.
        metric : str
            Distance metric to use for the tree.
        verbose : bool
            Print model details after building.
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
        )

        if verbose:
            print(f"KNN built: n_neighbors={n_neighbors}, weights={weights}, metric={metric}")

        return self

    # ------------------------------------------------------------------ #
    #  Train
    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train):
        """Train the model."""
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
    def save(self, filepath='knn_model.pkl'):
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
    def load_and_preprocess_data(filepath='Data/Features/emg_final_4.csv'):
        """Load a feature CSV and return (X, y) numpy arrays."""
        import pandas as pd

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None, None

        feature_cols = ['filt_AR_2', 'filt_AR_3', 'env_AR_3', 'env_WAMP']
        X = df[feature_cols].values
        y = df['Output'].values
        return X, y

    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=13):
        """Split arrays into train/test sets."""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


SELECTED_FEATURES = ['filt_AR_2', 'filt_AR_3', 'env_AR_3', 'env_WAMP']
CLASS_NAMES = ["Off", "Click"]


def log_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('KNN – Confusion Matrix')
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()
    fig.savefig(tmp_path)
    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path='graphs')
    os.remove(tmp_path)
    print("Confusion matrix logged to MLflow.")


# ====================================================================== #
#  Main
# ====================================================================== #
if __name__ == "__main__":
    print("=" * 60)
    print(f"  EMG KNN  |  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    file_path = sys.argv[1] if len(sys.argv) > 1 else 'Data/Features/emg_final_4.csv'

    # --- Model / training parameters ---
    n_neighbors = 14
    weights = 'distance'
    metric = 'minkowski'
    test_size = 0.2
    random_state = 13

    # --- MLflow setup ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EMG")

    print(f"Loading data from {file_path}...")
    X, y = EMGKNN.load_and_preprocess_data(file_path)

    if X is not None and y is not None:
        print(f"Data loaded. Shape: X={X.shape}, y={y.shape}")
        print(f"Class distribution (before SMOTE): {dict(zip(*np.unique(y, return_counts=True)))}")

        X_train, X_test, y_train, y_test = EMGKNN.split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        smote = BorderlineSMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE – Train: {X_train.shape}")
        print(f"Class distribution (after SMOTE): {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test: {X_test.shape}")

        run_name = f"KNN-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        with mlflow.start_run(run_name=run_name):

            mlflow.set_tags({
                "model_type": "KNeighborsClassifier",
                "feature_source": "emg_final_4",
                "balancing": "BorderlineSMOTE",
            })

            mlflow.log_params({
                "data_file": file_path,
                "input_dim": X_train.shape[1],
                "num_samples": X_train.shape[0],
                "test_size": test_size,
                "random_state": random_state,
                "selected_features": str(SELECTED_FEATURES),
                "num_features": len(SELECTED_FEATURES),
                "balancing": "BorderlineSMOTE",
                "model_type": "KNeighborsClassifier",
                "n_neighbors": n_neighbors,
                "weights": weights,
                "metric": metric,
            })

            print("Building model...")
            knn = EMGKNN(input_dim=X_train.shape[1])
            knn.build(n_neighbors=n_neighbors, weights=weights, metric=metric, verbose=True)

            print("Training model...")
            knn.train(X_train, y_train)

            # --- Evaluate ---
            test_probs = knn.model.predict_proba(X_test)[:, 1]
            y_pred = (test_probs >= 0.5).astype(int)
            test_acc = accuracy_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred, average=None, labels=[0, 1], zero_division=0)
            test_auc = roc_auc_score(y_test, test_probs)

            train_probs = knn.model.predict_proba(X_train)[:, 1]
            y_train_pred = (train_probs >= 0.5).astype(int)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred, average=None, labels=[0, 1], zero_division=0)
            train_auc = roc_auc_score(y_train, train_probs)

            print(f"\n  Test  Accuracy: {test_acc * 100:.2f}%  |  AUC-ROC: {test_auc:.4f}")
            print(f"  Train Accuracy: {train_acc * 100:.2f}%  |  AUC-ROC: {train_auc:.4f}")
            for cls_idx in range(2):
                print(f"  Test  Recall ({cls_idx}): {test_recall[cls_idx] * 100:.2f}%")
            print("\nClassification Report (test):")
            print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
            print("Confusion Matrix (test):")
            print(confusion_matrix(y_test, y_pred))

            # --- Log metrics ---
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_auc_roc", test_auc)
            mlflow.log_metric("train_auc_roc", train_auc)
            for cls_idx in range(2):
                mlflow.log_metric(f"test_recall_class_{cls_idx}", test_recall[cls_idx])
                mlflow.log_metric(f"train_recall_class_{cls_idx}", train_recall[cls_idx])

            log_confusion_matrix(y_test, y_pred)

            knn.save()
            mlflow.log_artifact("knn_model.pkl")

            print(f"\n  MLflow run '{run_name}' logged to experiment 'EMG'.")
