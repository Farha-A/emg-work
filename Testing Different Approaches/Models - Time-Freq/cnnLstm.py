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
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout,
    BatchNormalization, Input, TimeDistributed, Flatten,
)
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, recall_score,
)
from imblearn.over_sampling import SMOTE
from scipy.signal import stft as scipy_stft
import mlflow


class EMGCNNLSTM:
    """CNN-LSTM hybrid for EMG binary classification on STFT spectrograms.

    Pipeline
    --------
    1.  Load raw EMG stream CSV.
    2.  Group by (session_id, level_number) and create windows of 50 samples
        with **50 % overlap**.
    3.  Map labels:  levels {2, 4} → 1 (click), {1, 3, 5, 6, 7} → 0.
    4.  Compute STFT on each window → magnitude spectrogram.
    5.  Z-score normalise each channel globally.
    6.  Apply **SMOTE** to oversample class 1 on the training split.
    7.  Feed spectrograms into a CNN (spatial feature extraction)
        followed by an LSTM (temporal modelling) and a sigmoid head.
    """

    def __init__(self, model=None):
        self.model = model

    # ------------------------------------------------------------------ #
    #  Label mapping
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

    # ------------------------------------------------------------------ #
    #  STFT helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_stft(signal_1d, fs=100, nperseg=16, noverlap=8):
        """Compute STFT magnitude for a 1-D signal.

        Returns the magnitude spectrogram (freq_bins, time_frames).
        """
        _, _, Zxx = scipy_stft(signal_1d, fs=fs, nperseg=nperseg,
                               noverlap=noverlap)
        return np.abs(Zxx)  # magnitude

    # ------------------------------------------------------------------ #
    #  Data loading & STFT preprocessing
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_and_preprocess_data(csv_path, window_size=50, overlap=0.5,
                                  stft_nperseg=16, stft_noverlap=8, fs=100,
                                  channel_mode='both'):
        """Load EMG CSV, window with overlap, compute STFT spectrograms.

        Parameters
        ----------
        csv_path : str
            Path to the EMG stream CSV.
        window_size : int
            Number of raw samples per window (default 50).
        overlap : float
            Fraction of overlap between consecutive windows (0.5 = 50 %).
        stft_nperseg : int
            STFT segment length.
        stft_noverlap : int
            STFT overlap in samples.
        fs : int
            Sampling frequency (Hz) assumed for STFT.
        channel_mode : str
            Which signal channels to use for STFT.
            ``'both'``     → 2-channel (filtered + envelope)
            ``'filtered'`` → 1-channel (filtered only)
            ``'envelope'`` → 1-channel (envelope only)

        Returns
        -------
        X : np.ndarray
            shape (n_samples, freq_bins, time_frames, n_channels)
            n_channels is 2 for 'both', 1 for 'filtered' or 'envelope'.
        y : np.ndarray
            shape (n_samples,)
        """
        assert channel_mode in ('both', 'filtered', 'envelope'), \
            f"channel_mode must be 'both', 'filtered', or 'envelope', got '{channel_mode}'"
        df = pd.read_csv(csv_path)

        # Parse the tuple string into two numeric columns
        df['parsed_value'] = df['value'].apply(ast.literal_eval)
        df['filtered_value'] = df['parsed_value'].apply(lambda v: v[0])
        df['envelope_value'] = df['parsed_value'].apply(lambda v: v[1])

        step = int(window_size * (1 - overlap))  # hop = 25 for 50 % overlap

        spectrograms = []
        labels = []

        for (sid, lvl), grp in df.groupby(['session_id', 'level_number']):
            label = EMGCNNLSTM.get_output_label(lvl)
            if label == -1:
                continue

            filtered = grp['filtered_value'].values
            envelope = grp['envelope_value'].values

            # Sliding window with overlap
            for start in range(0, len(filtered) - window_size + 1, step):
                win_filt = filtered[start:start + window_size]
                win_env  = envelope[start:start + window_size]

                # STFT on selected channel(s) → magnitude spectrogram
                if channel_mode == 'both':
                    mag_filt = EMGCNNLSTM.compute_stft(
                        win_filt, fs=fs, nperseg=stft_nperseg,
                        noverlap=stft_noverlap,
                    )
                    mag_env = EMGCNNLSTM.compute_stft(
                        win_env, fs=fs, nperseg=stft_nperseg,
                        noverlap=stft_noverlap,
                    )
                    spectrogram = np.stack([mag_filt, mag_env], axis=-1)
                elif channel_mode == 'filtered':
                    mag_filt = EMGCNNLSTM.compute_stft(
                        win_filt, fs=fs, nperseg=stft_nperseg,
                        noverlap=stft_noverlap,
                    )
                    spectrogram = mag_filt[..., np.newaxis]  # (freq, time, 1)
                else:  # 'envelope'
                    mag_env = EMGCNNLSTM.compute_stft(
                        win_env, fs=fs, nperseg=stft_nperseg,
                        noverlap=stft_noverlap,
                    )
                    spectrogram = mag_env[..., np.newaxis]   # (freq, time, 1)

                spectrograms.append(spectrogram)
                labels.append(label)

        X = np.array(spectrograms, dtype='float32')
        y = np.array(labels, dtype='float32')

        return X, y

    # ------------------------------------------------------------------ #
    #  Z-score normalisation (per channel, computed on train set)
    # ------------------------------------------------------------------ #
    @staticmethod
    def zscore_normalize(X_train, X_test):
        """Apply z-score normalisation per channel.

        Statistics are computed on X_train and applied to both splits.

        Parameters
        ----------
        X_train, X_test : np.ndarray
            Shape (n, freq_bins, time_frames, channels).

        Returns
        -------
        X_train_norm, X_test_norm, means, stds
        """
        n_channels = X_train.shape[-1]
        means = np.zeros(n_channels)
        stds = np.zeros(n_channels)

        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()

        for ch in range(n_channels):
            mu = X_train[:, :, :, ch].mean()
            sigma = X_train[:, :, :, ch].std()
            if sigma == 0:
                sigma = 1.0  # avoid division by zero
            means[ch] = mu
            stds[ch] = sigma

            X_train_norm[:, :, :, ch] = (X_train[:, :, :, ch] - mu) / sigma
            X_test_norm[:, :, :, ch]  = (X_test[:, :, :, ch] - mu) / sigma

        return X_train_norm, X_test_norm, means, stds

    # ------------------------------------------------------------------ #
    #  SMOTE oversampling
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_smote(X, y, random_state=13):
        """Apply SMOTE to oversample the minority class (1).

        SMOTE works on 2-D arrays, so spectrograms are flattened first
        and reshaped back afterwards.

        Returns
        -------
        X_resampled, y_resampled  (same shape schema as inputs)
        """
        original_shape = X.shape[1:]  # (freq_bins, time_frames, channels)
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_flat, y)

        X_res = X_res.reshape(-1, *original_shape).astype('float32')
        y_res = y_res.astype('float32')

        print(f"SMOTE: 0 → {int((y == 0).sum())} → {int((y_res == 0).sum())}, "
              f"1 → {int((y == 1).sum())} → {int((y_res == 1).sum())}")
        return X_res, y_res

    # ------------------------------------------------------------------ #
    #  Loss
    # ------------------------------------------------------------------ #
    @staticmethod
    def focal_loss(gamma=2.0, alpha=0.25):
        """Binary focal loss (Lin et al., 2017)."""
        def _focal_loss(y_true, y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            focal_weight = alpha_t * K.pow(1.0 - p_t, gamma)
            return K.mean(focal_weight * bce)
        return _focal_loss

    # ------------------------------------------------------------------ #
    #  Build CNN-LSTM
    # ------------------------------------------------------------------ #
    def build(self, input_shape, cnn_filters=(32, 64), lstm_units=64,
              dropout=0.3, learning_rate=1e-3,
              focal_gamma=2.0, focal_alpha=0.25, verbose=False):
        """Construct the CNN-LSTM hybrid.

        Architecture
        ------------
        The spectrogram (freq_bins, time_frames, channels) is treated
        as an image.  Two Conv2D + MaxPool blocks extract spatial
        features, which are then reshaped into a sequence and fed
        into an LSTM for temporal modelling.  A Dense(1, sigmoid) head
        performs binary classification.

        Parameters
        ----------
        input_shape : tuple
            (freq_bins, time_frames, channels)  e.g. (9, 7, 2)
        cnn_filters : tuple[int]
            Number of filters for each Conv2D block.
        lstm_units : int
            Hidden units in the LSTM layer.
        dropout : float
            Dropout rate after CNN and LSTM.
        learning_rate : float
            Adam learning rate.
        focal_gamma, focal_alpha : float
            Focal-loss hyper-parameters.
        verbose : bool
            Print model summary.
        """
        freq_bins, time_frames, channels = input_shape

        m = Sequential([
            Input(shape=input_shape),

            # --- CNN block 1 ---
            Conv2D(cnn_filters[0], (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), padding='same'),
            Dropout(dropout),

            # --- CNN block 2 ---
            Conv2D(cnn_filters[1], (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), padding='same'),
            Dropout(dropout),

            # --- Reshape to sequence for LSTM ---
            # After two MaxPool(2,2) with 'same' padding the spatial dims
            # are reduced.  We collapse the frequency axis into features
            # and keep the (reduced) time axis as the sequence dimension.
            Reshape((-1, self._compute_collapsed_features(
                freq_bins, time_frames, channels, cnn_filters))),

            # --- LSTM ---
            LSTM(lstm_units),
            Dropout(dropout),

            # --- Output ---
            Dense(1, activation='sigmoid'),
        ])

        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.focal_loss(gamma=focal_gamma, alpha=focal_alpha),
            metrics=['accuracy'],
        )

        if verbose:
            m.summary()

        self.model = m
        return self

    @staticmethod
    def _compute_collapsed_features(freq_bins, time_frames, channels,
                                     cnn_filters):
        """Work out the feature-dim after Reshape.

        After two MaxPool(2,2) with 'same' padding:
            reduced_freq  = ceil(ceil(freq_bins / 2) / 2)
            reduced_time  = ceil(ceil(time_frames / 2) / 2)
        Reshape target: (reduced_time, reduced_freq * last_filter_count)
        But because we use Reshape((-1, feat)) the time dim is inferred.
        We return  reduced_freq * last_filter_count.
        """
        import math
        rf = math.ceil(math.ceil(freq_bins / 2) / 2)
        return rf * cnn_filters[-1]

    # ------------------------------------------------------------------ #
    #  Train
    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train, epochs=50, batch_size=16):
        """Train the CNN-LSTM. Returns the Keras history object."""
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
        print(classification_report(y_test, y_pred,
                                     target_names=["Off", "Click"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def save(self, filepath='cnn_lstm_model.h5'):
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
    #  Split
    # ------------------------------------------------------------------ #
    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=13):
        """Split arrays into train/test sets."""
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)

    # ------------------------------------------------------------------ #
    #  MLflow helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def log_training_graphs(history):
        """Log accuracy & loss curves directly to MLflow (no local file)."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'],   label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('CNN-LSTM – Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'],     label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('CNN-LSTM – Loss')
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
        ax.set_title('CNN-LSTM – Confusion Matrix')
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
            'Data', 'Stream', 'emg_stream_combined.csv',
        )
    )

    # --- Hyper-parameters ---
    window_size    = 50       # raw samples per window
    overlap        = 0.7      # 50 % overlap
    stft_nperseg   = 16       # STFT segment length
    stft_noverlap  = 10        # STFT overlap
    fs             = 100      # assumed sampling rate (Hz)

    # --- Channel mode toggle ---
    # Options: 'both' (filtered + envelope), 'filtered', 'envelope'
    channel_mode   = 'envelope'

    cnn_filters    = (32, 64)
    lstm_units     = 128
    dropout        = 0.3
    focal_gamma    = 3.5
    focal_alpha    = 0.65
    learning_rate  = 1e-4
    epochs         = 500
    batch_size     = 32
    test_size      = 0.2
    random_state   = 13

    # --- MLflow setup (use the existing DB in the parent Cleaning folder) ---
    cleaning_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mlflow_db = os.path.join(cleaning_dir, 'mlflow.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("EMG")

    # ------------------------------------------------------------------ #
    #  1.  Load & STFT
    # ------------------------------------------------------------------ #
    print(f"Loading data from {csv_path} ...")
    X, y = EMGCNNLSTM.load_and_preprocess_data(
        csv_path,
        window_size=window_size,
        overlap=overlap,
        stft_nperseg=stft_nperseg,
        stft_noverlap=stft_noverlap,
        fs=fs,
        channel_mode=channel_mode,
    )
    print(f"Channel mode: {channel_mode}")
    print(f"STFT data ready.  X shape: {X.shape},  y shape: {y.shape}")
    print(f"Class distribution: 0 → {int((y == 0).sum())}, "
          f"1 → {int((y == 1).sum())}")

    # ------------------------------------------------------------------ #
    #  2.  SMOTE on full dataset (before split)
    # ------------------------------------------------------------------ #
    X, y = EMGCNNLSTM.apply_smote(X, y, random_state=random_state)

    # ------------------------------------------------------------------ #
    #  3.  Train / Test split
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = EMGCNNLSTM.split_data(
        X, y, test_size=test_size, random_state=random_state,
    )

    # ------------------------------------------------------------------ #
    #  4.  Z-score normalisation (fit on train, apply to both)
    # ------------------------------------------------------------------ #
    X_train, X_test, z_means, z_stds = EMGCNNLSTM.zscore_normalize(
        X_train, X_test,
    )
    print(f"Z-score normalisation applied.  "
          f"Means: {z_means},  Stds: {z_stds}")
    print(f"Train: {X_train.shape},  Test: {X_test.shape}")

    # ------------------------------------------------------------------ #
    #  5.  Build, train, evaluate  (all inside an MLflow run)
    # ------------------------------------------------------------------ #
    input_shape = X_train.shape[1:]  # (freq_bins, time_frames, channels)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "data_file": csv_path,
            "window_size": window_size,
            "overlap": overlap,
            "stft_nperseg": stft_nperseg,
            "stft_noverlap": stft_noverlap,
            "fs": fs,
            "input_shape": str(input_shape),
            "num_samples": X.shape[0],
            "test_size": test_size,
            "random_state": random_state,
            "model_type": "CNN-LSTM",
            "cnn_filters": str(cnn_filters),
            "lstm_units": lstm_units,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            "epochs": epochs,
            "batch_size": batch_size,
            "normalization": "z-score",
            "balancing": "SMOTE",
            "channel_mode": channel_mode,
        })

        # Build
        print("Building CNN-LSTM model ...")
        cnn_lstm = EMGCNNLSTM()
        cnn_lstm.build(
            input_shape=input_shape,
            cnn_filters=cnn_filters,
            lstm_units=lstm_units,
            dropout=dropout,
            learning_rate=learning_rate,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            verbose=True,
        )

        # Train
        print("Training ...")
        history = cnn_lstm.train(X_train, y_train, epochs=epochs,
                                 batch_size=batch_size)

        # Evaluate
        accuracy = cnn_lstm.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

        y_pred = (cnn_lstm.model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
        cnn_lstm.detailed_report(X_test, y_test)

        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy)

        train_acc = cnn_lstm.evaluate(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_acc)

        y_train_pred = (cnn_lstm.model.predict(X_train, verbose=0) > 0.5).astype(int).ravel()

        # Calculate recall for both classes
        test_recalls = recall_score(y_test, y_pred, average=None)
        train_recalls = recall_score(y_train, y_train_pred, average=None)

        print(f"Test Recall Class 0: {test_recalls[0] * 100:.2f}% | "
              f"Class 1: {test_recalls[1] * 100:.2f}%")

        mlflow.log_metric("test_recall_class_0", test_recalls[0])
        mlflow.log_metric("test_recall_class_1", test_recalls[1])
        mlflow.log_metric("train_recall_class_0", train_recalls[0])
        mlflow.log_metric("train_recall_class_1", train_recalls[1])

        # Log training graphs to MLflow (no local files)
        EMGCNNLSTM.log_training_graphs(history)

        # Log confusion matrix to MLflow (no local files)
        EMGCNNLSTM.log_confusion_matrix(y_test, y_pred)

        # Save model and log to MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'cnn_lstm_model.h5')
            cnn_lstm.save(model_path)
            mlflow.log_artifact(model_path)

        print(f"MLflow run logged to experiment 'EMG'")
