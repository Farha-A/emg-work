"""
NN-3Class Model
================
Trains a neural network for 3-class EMG classification using 4 features:
  filt_AR_2, filt_AR_3, env_AR_3, env_WAMP

Level mapping (level_number -> class label):
  2, 4 -> 1   (Max / Preference contraction)
  8    -> 2   (Double click — synthesized)
  1, 3, 5, 6, 7 -> 0

Data source: emg_stream_combined_with_double_click.csv
Uses SMOTE for class balancing.
Model details and metrics are logged to MLflow (experiment 'EMG-3').
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
from imblearn.over_sampling import BorderlineSMOTE
import mlflow

# Reproducibility
np.random.seed(13)
tf.random.set_seed(13)

# ---------------------------------------------------------------------------
# Paths  (relative to this script, matching the project layout)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(SCRIPT_DIR, os.pardir,
                            'Testing Different Approaches', 'Features Sets')

# Import shared helpers from all_features_sfs
sys.path.insert(0, os.path.normpath(FEATURES_DIR))
from all_features_sfs import (
    load_and_extract_features,
    build_model,
)

DATA_PATH = os.path.join(
    SCRIPT_DIR, os.pardir,
    'Data', 'Stream', 'emg_stream_combined_with_double_click.csv'
)
# Hyperparameters are now hardcoded in the main() function

# ---------------------------------------------------------------------------
# Segment length
# ---------------------------------------------------------------------------
SEGMENT_LENGTH = 10

# ---------------------------------------------------------------------------
# Feature set to use
# ---------------------------------------------------------------------------
SELECTED_FEATURES = [
    'filt_AR_2',
    'filt_AR_3',
    'env_AR_3',
    'env_WAMP',
]

# ---------------------------------------------------------------------------
# 3-class label mapping:  level_number -> class
# ---------------------------------------------------------------------------
LABEL_MAP = {
    2: 1, 4: 1,
    8: 2,
    1: 0, 3: 0, 5: 0, 6: 0, 7: 0,
}
CLASS_NAMES = ["Other", "Click (Max/Pref)", "Double Click"]
NUM_CLASSES = 3


# ===========================  CUSTOM DATA LOADER  ============================

def load_and_extract_features_3class(file_path, segment_size=50):
    """
    Wraps the shared loader but overrides the label mapping to produce
    3-class labels instead of binary labels.
    """
    import pandas as pd
    import ast

    print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path)
    print(f"  Raw rows: {df.shape[0]}")

    # Parse the tuple-like 'value' column into two floats
    def parse_value(v):
        try:
            return ast.literal_eval(str(v))
        except (ValueError, SyntaxError):
            return (0.0, 0.0)

    parsed = df['value'].apply(parse_value)
    df['filtered'] = parsed.apply(lambda t: t[0])
    df['envelope'] = parsed.apply(lambda t: t[1])

    # Apply 3-class label mapping
    df['label'] = df['level_number'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Import feature extraction functions from all_features_sfs
    from all_features_sfs import calculate_emg_features, calculate_emg_spectral_features

    # Group by session + level to get contiguous signal chunks
    all_features = []
    for (sid, lvl), grp in df.groupby(['session_id', 'level_number'], sort=False):
        filt_vals = grp['filtered'].values
        env_vals = grp['envelope'].values
        output_label = grp['label'].iloc[0]

        n_samples = len(filt_vals)
        for start in range(0, n_samples, segment_size):
            filt_seg = filt_vals[start:start + segment_size]
            env_seg = env_vals[start:start + segment_size]

            if len(filt_seg) < 5:  # skip tiny tail segments
                continue

            combined = {}

            # --- Time-domain features ---
            filt_feats = calculate_emg_features(filt_seg)
            for k, v in filt_feats.items():
                combined[f'filt_{k}'] = v

            env_feats = calculate_emg_features(env_seg)
            for k, v in env_feats.items():
                combined[f'env_{k}'] = v

            # --- Frequency-domain features ---
            filt_spec = calculate_emg_spectral_features(filt_seg)
            for k, v in filt_spec.items():
                combined[f'filt_spec_{k}'] = v

            env_spec = calculate_emg_spectral_features(env_seg)
            for k, v in env_spec.items():
                combined[f'env_spec_{k}'] = v

            combined['Output'] = output_label
            all_features.append(combined)

    feat_df = pd.DataFrame(all_features)
    # Replace any inf/nan with 0
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.fillna(0, inplace=True)

    print(f"  Extracted {len(feat_df)} segments  |  "
          f"Features per segment: {len(feat_df.columns) - 1}")
    print(f"  Class distribution: {dict(feat_df['Output'].value_counts())}")
    return feat_df


# ===========================  3-CLASS MODEL BUILDER  =========================

def build_3class_model(input_dim, hp):
    """Build a Keras Sequential NN for 3-class classification."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(hp['hidden_layers']):
        model.add(Dense(hp['neurons'], activation='relu'))
        model.add(Dropout(hp['dropout']))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


# ===========================  MLflow HELPERS  ================================

def log_training_graphs(history):
    """Log accuracy & loss curves to MLflow as an artifact."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],   label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('NN-3Class – Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('NN-3Class – Loss')
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
                                  display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('NN-3Class – Confusion Matrix')
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
    print(f"  NN-3Class  |  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  1.  Hyperparameters
    # ------------------------------------------------------------------ #
    hp = {
        'hidden_layers': 2,
        'neurons': 64,
        'dropout': 0.2
    }
    print(f"  Hyperparameters: {hp}")

    # ------------------------------------------------------------------ #
    #  2.  Extract ALL features using segment_size=30 with 3-class labels
    # ------------------------------------------------------------------ #
    df = load_and_extract_features_3class(DATA_PATH, segment_size=SEGMENT_LENGTH)
    if df is None or df.empty:
        print("ERROR: no data extracted.")
        return

    # Verify requested features exist
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: missing features in data: {missing}")
        return

    # ------------------------------------------------------------------ #
    #  3.  Prepare X, y with only the selected features
    # ------------------------------------------------------------------ #
    X = df[SELECTED_FEATURES].values
    y = df['Output'].values

    print(f"\n  Selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    print(f"  Label mapping: {LABEL_MAP}")
    print(f"  Segment length: {SEGMENT_LENGTH}")
    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"  Class distribution (before SMOTE): {dict(zip(*np.unique(y, return_counts=True)))}")

    # ------------------------------------------------------------------ #
    #  4.  Train / Test split
    # ------------------------------------------------------------------ #
    test_size = 0.2
    random_state = 13
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ------------------------------------------------------------------ #
    #  5.  Apply SMOTE to training data
    # ------------------------------------------------------------------ #
    smote = BorderlineSMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE – Train: {X_train.shape}")
    print(f"  Class distribution (after SMOTE): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ------------------------------------------------------------------ #
    #  6.  MLflow setup
    # ------------------------------------------------------------------ #
    cleaning_dir = os.path.normpath(
        os.path.join(SCRIPT_DIR, os.pardir)
    )
    mlflow_db = os.path.join(cleaning_dir, 'mlflow.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("EMG-3")

    # ------------------------------------------------------------------ #
    #  7.  Build, train, evaluate – all inside an MLflow run
    # ------------------------------------------------------------------ #
    run_name = f"NN-3Class-DoubleClick-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name):

        mlflow.set_tags({
            "class_2_gesture": "double_click_level_8",
            "previous_class_2": "eyebrow_raise_level_6",
            "data_variant": "with_double_click",
            "notes": "Class 2 swapped from level 6 (Eyebrows raise) to level 8 (Double Click). Level 6 folded into class 0.",
        })

        # --- Log parameters ---
        mlflow.log_params({
            # Data parameters
            "data_file": DATA_PATH,
            "input_dim": X_train.shape[1],
            "num_samples": X_train.shape[0],
            "num_classes": NUM_CLASSES,
            "test_size": test_size,
            "random_state": random_state,
            "segment_length": int(SEGMENT_LENGTH),
            "label_mapping": str(LABEL_MAP),
            # Model parameters
            "hidden_layers": hp['hidden_layers'],
            "neurons": hp['neurons'],
            "dropout": hp['dropout'],
            "model_type": "NN-3Class",
            # Feature set
            "selected_features": str(SELECTED_FEATURES),
            "num_features": len(SELECTED_FEATURES),
            # Balancing
            "balancing": "BorderlineSMOTE",
            # Class identity
            "class_2_gesture": "double_click_level_8",
            "class_names": str(CLASS_NAMES),
        })

        # --- Build model ---
        print("\nBuilding model ...")
        model = build_3class_model(X_train.shape[1], hp)
        model.summary()

        # --- Train ---
        print("\nTraining ...")
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            epochs=50, batch_size=32, verbose=1,
            validation_split=0.2, callbacks=[es],
        )

        # --- Evaluate on test set ---
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        test_acc = accuracy_score(y_test, y_pred)
        test_recall_per_class = recall_score(
            y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )

        print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
        for cls_idx in range(NUM_CLASSES):
            print(f"  Recall ({cls_idx})   : {test_recall_per_class[cls_idx] * 100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                     target_names=CLASS_NAMES,
                                     zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # --- Evaluate on train set ---
        y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_recall_per_class = recall_score(
            y_train, y_train_pred, average=None, labels=[0, 1, 2], zero_division=0
        )

        # --- Log metrics ---
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_accuracy", train_acc)
        for cls_idx in range(NUM_CLASSES):
            mlflow.log_metric(f"test_recall_class_{cls_idx}", test_recall_per_class[cls_idx])
            mlflow.log_metric(f"train_recall_class_{cls_idx}", train_recall_per_class[cls_idx])

        # --- Log training graphs ---
        log_training_graphs(history)

        # --- Log confusion matrix ---
        log_confusion_matrix(y_test, y_pred)

        # --- Save model and log to MLflow ---
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'nn_3class_model.h5')
            model.save(model_path)
            mlflow.log_artifact(model_path)

        print(f"\n✓ MLflow run '{run_name}' logged to experiment 'EMG-3' "
              f"(class 2 = Double Click, segment_length={SEGMENT_LENGTH})")


if __name__ == '__main__':
    main()
