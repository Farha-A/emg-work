import sys
import os
import tempfile
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
from imblearn.over_sampling import BorderlineSMOTE
from model import EMGModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, 'Data', 'Stream', 'emg_stream_combined.csv')
SEGMENT_LENGTH = 30
SELECTED_FEATURES = ['filt_AR_2', 'filt_AR_3', 'env_AR_3', 'env_WAMP']
LABEL_MAP = {2: 1, 4: 1, 1: 0, 3: 0, 5: 0, 6: 0, 7: 0}
CLASS_NAMES = ["Other", "Click (Max/Pref)"]


def log_training_graphs(history):
    """Log accuracy & loss curves to MLflow as artifacts."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('EMG Model – Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('EMG Model – Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()
    plt.savefig(tmp_path)
    plt.close()
    mlflow.log_artifact(tmp_path, artifact_path='graphs')
    os.remove(tmp_path)
    print("Training graphs logged to MLflow.")


def log_confusion_matrix(y_true, y_pred):
    """Log a confusion-matrix figure to MLflow as an artifact."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('EMG Model – Confusion Matrix')
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()
    fig.savefig(tmp_path)
    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path='graphs')
    os.remove(tmp_path)
    print("Confusion matrix logged to MLflow.")


if __name__ == "__main__":
    print("=" * 60)
    print(f"  EMG Binary Model  |  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # --- Model / training parameters ---
    hidden_layers = 2
    neurons = 8
    dropout = 0.4100
    epochs = 50
    batch_size = 32
    test_size = 0.2
    random_state = 13

    # --- Load & extract features from raw stream ---
    df = EMGModel.load_and_extract_stream_features(DATA_PATH, segment_size=SEGMENT_LENGTH)
    if df is None or df.empty:
        print("ERROR: no data extracted.")
        sys.exit(1)

    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: missing features in data: {missing}")
        sys.exit(1)

    X = df[SELECTED_FEATURES].values
    y = df['Output'].values

    print(f"\n  Selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    print(f"  Label mapping: {LABEL_MAP}")
    print(f"  Segment length: {SEGMENT_LENGTH}")
    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"  Class distribution (before SMOTE): {dict(zip(*np.unique(y, return_counts=True)))}")

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # --- Balance training set ---
    smote = BorderlineSMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE – Train: {X_train.shape}")
    print(f"  Class distribution (after SMOTE): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # --- MLflow setup ---
    mlflow_db = os.path.join(ROOT_DIR, 'mlflow.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("EMG")

    run_name = f"EMG-Binary-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name):

        mlflow.set_tags({
            "model_type": "binary_nn",
            "feature_source": "emg_stream_combined",
            "balancing": "BorderlineSMOTE",
        })

        mlflow.log_params({
            "data_file": DATA_PATH,
            "input_dim": X_train.shape[1],
            "num_samples": X_train.shape[0],
            "test_size": test_size,
            "random_state": random_state,
            "segment_length": int(SEGMENT_LENGTH),
            "label_mapping": str(LABEL_MAP),
            "hidden_layers": hidden_layers,
            "neurons": neurons,
            "dropout": dropout,
            "loss_function": "binary_crossentropy",
            "optimizer": "adam",
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stop_monitor": "accuracy",
            "early_stop_patience": 5,
            "validation_split": 0.2,
            "selected_features": str(SELECTED_FEATURES),
            "num_features": len(SELECTED_FEATURES),
            "balancing": "BorderlineSMOTE",
        })

        print("\nBuilding model ...")
        emg = EMGModel(input_dim=X_train.shape[1])
        emg.build(
            hidden_layers=hidden_layers,
            neurons=neurons,
            dropout=dropout,
            verbose=True,
        )

        print("\nTraining ...")
        history = emg.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # --- Evaluate on test set ---
        test_probs = emg.model.predict(X_test, verbose=0).ravel()
        y_pred = (test_probs >= 0.5).astype(int)
        test_acc = accuracy_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred, average=None, labels=[0, 1], zero_division=0)
        test_auc = roc_auc_score(y_test, test_probs)

        # --- Evaluate on train set ---
        train_probs = emg.model.predict(X_train, verbose=0).ravel()
        y_train_pred = (train_probs >= 0.5).astype(int)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred, average=None, labels=[0, 1], zero_division=0)
        train_auc = roc_auc_score(y_train, train_probs)

        print(f"\n  Test  Accuracy : {test_acc * 100:.2f}%  |  AUC-ROC: {test_auc:.4f}")
        print(f"  Train Accuracy : {train_acc * 100:.2f}%  |  AUC-ROC: {train_auc:.4f}")
        for cls_idx in range(2):
            print(f"  Test  Recall ({cls_idx})  : {test_recall[cls_idx] * 100:.2f}%")
        print("\nClassification Report (test):")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
        print("Confusion Matrix (test):")
        print(confusion_matrix(y_test, y_pred))

        # --- Log final metrics ---
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_auc_roc", test_auc)
        mlflow.log_metric("train_auc_roc", train_auc)
        for cls_idx in range(2):
            mlflow.log_metric(f"test_recall_class_{cls_idx}", test_recall[cls_idx])
            mlflow.log_metric(f"train_recall_class_{cls_idx}", train_recall[cls_idx])

        # --- Log per-epoch metrics ---
        for epoch_idx in range(len(history.history['loss'])):
            mlflow.log_metrics({
                "train_loss": history.history['loss'][epoch_idx],
                "train_accuracy": history.history['accuracy'][epoch_idx],
                "val_loss": history.history['val_loss'][epoch_idx],
                "val_accuracy": history.history['val_accuracy'][epoch_idx],
            }, step=epoch_idx)

        log_training_graphs(history)
        log_confusion_matrix(y_test, y_pred)

        emg.save()
        mlflow.log_artifact("bg_model.h5")

        print(f"\n  MLflow run '{run_name}' logged to experiment 'EMG'.")
