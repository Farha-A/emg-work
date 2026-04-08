import sys
import os

# Add parent directory to sys.path to allow imports from the main directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import ast
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score
import mlflow


class EMGLSTM:
    """Wraps a small Keras LSTM model for EMG binary classification.

    The model is fed raw (filtered_value, envelope_value) sequences
    derived from the 'value' column of the EMG stream CSV.  Each sequence
    corresponds to one (session_id, level_number) group.

    Mirrors the EMGModel API: build, train, evaluate, predict, save, load.
    """

    def __init__(self, model=None):
        self.model = model

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #
    def build(self, max_seq_len, n_features=2, lstm_units=32, dropout=0.3,
              verbose=False):
        """Construct a small LSTM network.

        Parameters
        ----------
        max_seq_len : int
            Length to which every sequence is padded/truncated.
        n_features : int
            Number of features per timestep (default 2: filtered, envelope).
        lstm_units : int
            Number of LSTM units (kept small for a lightweight model).
        dropout : float
            Dropout rate applied after the LSTM layer.
        verbose : bool
            Print model summary after building.
        """
        m = Sequential([
            Input(shape=(max_seq_len, n_features)),
            Masking(mask_value=0.0),       # ignore padded timesteps
            LSTM(lstm_units),
            Dropout(dropout),
            Dense(1, activation='sigmoid'),
        ])
        m.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        if verbose:
            m.summary()

        self.model = m
        self.max_seq_len = max_seq_len
        self.n_features = n_features
        return self

    # ------------------------------------------------------------------ #
    #  Train
    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train, epochs=50, batch_size=16):
        """Train the LSTM. Returns the Keras history object."""
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True,
        )
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop],
            validation_split=0.2,
        )
        return history

    # ------------------------------------------------------------------ #
    #  Evaluate / Predict
    # ------------------------------------------------------------------ #
    def evaluate(self, X_test, y_test):
        """Return accuracy on the test set."""
        _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return accuracy

    def predict(self, X):
        """Return a boolean prediction (threshold 0.5)."""
        value = self.model.predict(X, verbose=0)[0][0]
        return value > 0.5

    def detailed_report(self, X_test, y_test):
        """Print classification report and confusion matrix."""
        y_pred = (self.model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Off", "Click"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def save(self, filepath='lstm_model.h5'):
        """Persist model to disk."""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load a saved model from *filepath* and return an instance."""
        if not os.path.exists(filepath):
            print(f"Error: Model '{filepath}' not found.")
            return None
        model = load_model(filepath)
        return cls(model=model)

    # ------------------------------------------------------------------ #
    #  Data helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_output_label(level_number):
        """Map level_number to binary label.

        Swallow levels (2, 4) → 1, others (1, 3, 5, 6, 7) → 0.
        Returns -1 for unexpected values.
        """
        if level_number in [2, 4]:
            return 1
        elif level_number in [1, 3, 5, 6, 7]:
            return 0
        return -1

    @staticmethod
    def load_and_preprocess_data(csv_path, max_seq_len=None):
        """Load the raw EMG stream CSV and build padded sequences.

        Each (session_id, level_number) group is split into
        non-overlapping chunks of *max_seq_len* timesteps.  If a
        sequence is shorter than *max_seq_len* it is zero-padded.  If
        it is longer it produces multiple samples (all sharing the
        same label).

        Parameters
        ----------
        csv_path : str
            Path to the EMG stream CSV.
        max_seq_len : int or None
            Fixed sequence length.  If ``None`` the longest sequence
            in the data is used (no splitting).

        Returns
        -------
        X : np.ndarray   – shape (n_samples, max_seq_len, 2)
        y : np.ndarray   – shape (n_samples,)
        max_seq_len : int
        """
        df = pd.read_csv(csv_path)

        # Parse the tuple string into two numeric columns
        df['parsed_value'] = df['value'].apply(ast.literal_eval)
        df['filtered_value'] = df['parsed_value'].apply(lambda v: v[0])
        df['envelope_value'] = df['parsed_value'].apply(lambda v: v[1])

        raw_sequences = []
        raw_labels = []

        for (sid, lvl), grp in df.groupby(['session_id', 'level_number']):
            label = EMGLSTM.get_output_label(lvl)
            if label == -1:
                continue  # skip unexpected levels

            # Build (timesteps, 2) array using both value columns
            seq = grp[['filtered_value', 'envelope_value']].values
            raw_sequences.append(seq)
            raw_labels.append(label)

        if max_seq_len is None:
            max_seq_len = max(len(s) for s in raw_sequences)

        # Split long sequences into non-overlapping chunks of max_seq_len
        chunks = []
        chunk_labels = []
        for seq, label in zip(raw_sequences, raw_labels):
            for start in range(0, len(seq), max_seq_len):
                chunk = seq[start : start + max_seq_len]
                chunks.append(chunk)
                chunk_labels.append(label)

        # Pad short chunks to max_seq_len
        X = pad_sequences(
            chunks,
            maxlen=max_seq_len,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0,
        )
        y = np.array(chunk_labels, dtype='float32')

        return X, y, max_seq_len

    @staticmethod
    def upsample_minority(X, y):
        """Duplicate minority-class (label=1) sequences to match majority count."""
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        n_majority = len(idx_0)
        n_minority = len(idx_1)

        if n_minority == 0 or n_minority >= n_majority:
            return X, y

        # Repeat minority indices to reach majority count
        repeat_times = n_majority // n_minority
        remainder = n_majority % n_minority
        upsampled_idx = np.concatenate([
            np.tile(idx_1, repeat_times),
            np.random.RandomState(13).choice(idx_1, size=remainder, replace=False)
        ])

        X_up = np.concatenate([X[idx_0], X[upsampled_idx]], axis=0)
        y_up = np.concatenate([y[idx_0], y[upsampled_idx]], axis=0)

        # Shuffle
        perm = np.random.RandomState(13).permutation(len(y_up))
        print(f"Upsampled: 0 → {len(idx_0)}, 1 → {n_minority} ⇒ "
              f"0 → {int((y_up == 0).sum())}, 1 → {int((y_up == 1).sum())}")
        return X_up[perm], y_up[perm]

    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=13):
        """Split arrays into train/test sets."""
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)

    @staticmethod
    def log_training_graphs(history):
        """Log accuracy & loss curves directly to MLflow (no local file)."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'],   label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('LSTM – Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'],     label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('LSTM – Loss')
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

    @staticmethod
    def log_confusion_matrix(y_true, y_pred):
        """Log a confusion-matrix figure directly to MLflow (no local file)."""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Off", "Click"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title('LSTM – Confusion Matrix')
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
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Data', 'Stream', 'emg_stream_f.csv',
        )
    )

    # --- Hyper-parameters (small LSTM) ---
    lstm_units = 16
    dropout = 0.4
    epochs = 200
    batch_size = 16
    test_size = 0.2
    random_state = 13
    max_seq_len = 50

    # --- MLflow setup (use the existing DB in the parent Cleaning folder) ---
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow_db = os.path.join(parent_dir, 'mlflow.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("EMG")

    print(f"Loading data from {csv_path} ...")
    X, y, _ = EMGLSTM.load_and_preprocess_data(csv_path, max_seq_len=max_seq_len)
    print(f"Data loaded.  X shape: {X.shape},  y shape: {y.shape},  "
          f"max_seq_len: {max_seq_len}")
    print(f"Class distribution: 0 → {int((y == 0).sum())}, "
          f"1 → {int((y == 1).sum())}")

    X_train, X_test, y_train, y_test = EMGLSTM.split_data(
        X, y, test_size=test_size, random_state=random_state,
    )

    # Upsample minority class (1) in training set only
    X_train, y_train = EMGLSTM.upsample_minority(X_train, y_train)
    print(f"Train: {X_train.shape},  Test: {X_test.shape}")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "data_file": csv_path,
            "max_seq_len": max_seq_len,
            "n_features": 2,
            "num_sequences": X.shape[0],
            "test_size": test_size,
            "random_state": random_state,
            "model_type": "LSTM",
            "lstm_units": lstm_units,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
        })

        # Build
        print("Building LSTM model ...")
        lstm = EMGLSTM()
        lstm.build(
            max_seq_len=max_seq_len,
            n_features=2,
            lstm_units=lstm_units,
            dropout=dropout,
            verbose=True,
        )

        # Train
        print("Training ...")
        history = lstm.train(X_train, y_train, epochs=epochs,
                             batch_size=batch_size)

        # Evaluate
        accuracy = lstm.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

        y_pred = (lstm.model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
        lstm.detailed_report(X_test, y_test)

        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy)

        train_acc = lstm.evaluate(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_acc)

        y_train_pred = (lstm.model.predict(X_train, verbose=0) > 0.5).astype(int).ravel()
        
        # Calculate recall for both classes
        test_recalls = recall_score(y_test, y_pred, average=None)
        train_recalls = recall_score(y_train, y_train_pred, average=None)

        print(f"Test Recall Class 0: {test_recalls[0] * 100:.2f}% | Class 1: {test_recalls[1] * 100:.2f}%")

        mlflow.log_metric("test_recall_class_0", test_recalls[0])
        mlflow.log_metric("test_recall_class_1", test_recalls[1])
        mlflow.log_metric("train_recall_class_0", train_recalls[0])
        mlflow.log_metric("train_recall_class_1", train_recalls[1])

        # Log training graphs to MLflow (no local files)
        EMGLSTM.log_training_graphs(history)

        # Log confusion matrix to MLflow (no local files)
        EMGLSTM.log_confusion_matrix(y_test, y_pred)

        # Save model and log to MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'lstm_model.h5')
            lstm.save(model_path)
            mlflow.log_artifact(model_path)

        print(f"MLflow run logged to experiment 'EMG'")
