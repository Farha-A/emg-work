"""
NN-Time-Freq Model
==================
Trains a neural network on a fixed 5-feature set (4 time-domain + 1 freq-domain):
  Time-Domain  : env_VAR, filt_AAC, filt_DASDV, filt_SSC
  Freq-Domain  : filt_spec_TTP

Data cleaning, feature extraction, upsampling, and NN hyperparameters are
identical to the all_features_sfs.py pipeline.  Model details and metrics
are logged to MLflow (tagged as "NN-Time-Freq").
"""

import sys
import os
import tempfile
import datetime

# Add parent directory to sys.path (consistent with other model scripts)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow

# Reproducibility
np.random.seed(13)
tf.random.set_seed(13)

# ---------------------------------------------------------------------------
# Paths  (relative to this script, matching the project layout)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'Features Sets')

# Import the shared feature-extraction and data helpers from all_features_sfs
sys.path.insert(0, os.path.normpath(FEATURES_DIR))
from all_features_sfs import (
    load_and_extract_features,
    upsample_minority,
    parse_hyperparameters,
    build_model,
)

DATA_PATH = os.path.join(
    SCRIPT_DIR, os.pardir, os.pardir,
    'Data', 'Stream', 'emg_stream_combined.csv'
)
HYPERPARAMS_PATH = os.path.join(
    SCRIPT_DIR, os.pardir, os.pardir,
    'Optimizing Model Performance', 'hyperparameters.txt'
)

# ---------------------------------------------------------------------------
# Feature set to use
# ---------------------------------------------------------------------------
SELECTED_FEATURES = [
    # Time-Domain (4)
    'env_VAR',
    'filt_AAC',
    'filt_DASDV',
    'filt_SSC',
    # Frequency-Domain (1)
    'filt_spec_TTP',
]

TIME_DOMAIN_FEATURES  = ['env_VAR', 'filt_AAC', 'filt_DASDV', 'filt_SSC']
FREQ_DOMAIN_FEATURES  = ['filt_spec_TTP']


# ===========================  MLflow HELPERS  ================================

def log_training_graphs(history):
    """Log accuracy & loss curves to MLflow as an artifact."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],   label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('NN-Time-Freq – Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('NN-Time-Freq – Loss')
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Off", "Click"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('NN-Time-Freq – Confusion Matrix')
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()
    fig.savefig(tmp_path)
    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path='graphs')
    os.remove(tmp_path)
    print("Confusion matrix logged to MLflow.")


# ===========================  MAIN  ==========================================

def main():
    print("=" * 60)
    print(f"  NN-Time-Freq  |  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  1.  Hyperparameters
    # ------------------------------------------------------------------ #
    hp = parse_hyperparameters(HYPERPARAMS_PATH)

    # ------------------------------------------------------------------ #
    #  2.  Extract ALL features (time + freq), then select the 5 we need
    # ------------------------------------------------------------------ #
    df = load_and_extract_features(DATA_PATH)
    if df is None or df.empty:
        print("ERROR: no data extracted.")
        return

    # Verify requested features exist
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: missing features in data: {missing}")
        return

    # ------------------------------------------------------------------ #
    #  3.  Upsample minority class
    # ------------------------------------------------------------------ #
    df = upsample_minority(df, 'Output')

    # ------------------------------------------------------------------ #
    #  4.  Prepare X, y with only the selected features
    # ------------------------------------------------------------------ #
    X = df[SELECTED_FEATURES].values
    y = df['Output'].values

    print(f"\n  Selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    print(f"    Time-Domain  ({len(TIME_DOMAIN_FEATURES)}): {TIME_DOMAIN_FEATURES}")
    print(f"    Freq-Domain  ({len(FREQ_DOMAIN_FEATURES)}): {FREQ_DOMAIN_FEATURES}")
    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")

    # ------------------------------------------------------------------ #
    #  5.  Train / Test split
    # ------------------------------------------------------------------ #
    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ------------------------------------------------------------------ #
    #  6.  MLflow setup
    # ------------------------------------------------------------------ #
    cleaning_dir = os.path.normpath(
        os.path.join(SCRIPT_DIR, os.pardir, os.pardir)
    )
    mlflow_db = os.path.join(cleaning_dir, 'mlflow.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("EMG")

    # ------------------------------------------------------------------ #
    #  7.  Build, train, evaluate – all inside an MLflow run
    # ------------------------------------------------------------------ #
    with mlflow.start_run():
        # --- Log parameters ---
        mlflow.log_params({
            # Data parameters
            "data_file": DATA_PATH,
            "input_dim": X.shape[1],
            "num_samples": X.shape[0],
            "test_size": test_size,
            "random_state": random_state,
            # Model parameters
            "model_type": "NN-Time-Freq",
            "hidden_layers": hp['hidden_layers'],
            "neurons": hp['neurons'],
            "dropout": hp['dropout'],
            # Feature set
            "selected_features": str(SELECTED_FEATURES),
            "num_features": len(SELECTED_FEATURES),
            "time_domain_features": str(TIME_DOMAIN_FEATURES),
            "freq_domain_features": str(FREQ_DOMAIN_FEATURES),
            # Upsampling
            "balancing": "random_oversampling",
        })

        # --- Build model ---
        print("\nBuilding model ...")
        model = build_model(X_train.shape[1], hp)
        model.summary()

        # --- Train ---
        print("\nTraining ...")
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            epochs=50, batch_size=32, verbose=1,
            validation_split=0.2, callbacks=[es]
        )

        # --- Evaluate on test set ---
        y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).ravel()
        test_acc = accuracy_score(y_test, y_pred)
        test_rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        test_rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

        print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
        print(f"  Recall (0)   : {test_rec_0 * 100:.2f}%")
        print(f"  Recall (1)   : {test_rec_1 * 100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                     target_names=["Off", "Click"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # --- Evaluate on train set ---
        y_train_pred = (model.predict(X_train, verbose=0) >= 0.5).astype(int).ravel()
        train_acc = accuracy_score(y_train, y_train_pred)
        train_rec_0 = recall_score(y_train, y_train_pred, pos_label=0, zero_division=0)
        train_rec_1 = recall_score(y_train, y_train_pred, pos_label=1, zero_division=0)

        # --- Log metrics ---
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_recall_class_0", test_rec_0)
        mlflow.log_metric("test_recall_class_1", test_rec_1)
        mlflow.log_metric("train_recall_class_0", train_rec_0)
        mlflow.log_metric("train_recall_class_1", train_rec_1)

        # --- Log training graphs ---
        log_training_graphs(history)

        # --- Log confusion matrix ---
        log_confusion_matrix(y_test, y_pred)

        # --- Save model and log to MLflow ---
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'nn_time_freq_model.h5')
            model.save(model_path)
            mlflow.log_artifact(model_path)

        print(f"\n✓ MLflow run logged to experiment 'EMG'  (model_type=NN-Time-Freq)")


if __name__ == '__main__':
    main()
