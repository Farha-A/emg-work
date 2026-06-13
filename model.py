import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from config import INPUT_DIM


class EMGModel:
    """Wraps a Keras model for EMG binary classification.

    Handles building, training, evaluating, saving, loading,
    predicting, and fine-tuning.
    """

    DEFAULT_INPUT_DIM = INPUT_DIM

    def __init__(self, model=None, input_dim=None):
        self.model = model
        self.input_dim = input_dim or self.DEFAULT_INPUT_DIM

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #
    def build(self, hidden_layers=1, neurons=16, dropout=0.0, verbose=False):
        """Construct a dense neural network."""
        m = Sequential()
        m.add(Input(shape=(self.input_dim,)))

        for _ in range(hidden_layers):
            m.add(Dense(neurons, activation='relu'))
            if dropout > 0:
                m.add(tf.keras.layers.Dropout(dropout))

        m.add(Dense(1, activation='sigmoid'))
        m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if verbose:
            m.summary()

        self.model = m
        return self

    # ------------------------------------------------------------------ #
    #  Train / Fine-tune
    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train, epochs=5, batch_size=32):
        """Train (or continue training) the model. Returns the Keras history."""
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', patience=5, restore_best_weights=True
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

    def finetune(self, X_train, y_train, epochs=5, batch_size=32, lr=0.001):
        """Re-compile with a fresh Adam optimiser and train."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        return self.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # ------------------------------------------------------------------ #
    #  Evaluate / Predict
    # ------------------------------------------------------------------ #
    def evaluate(self, X_test, y_test):
        """Return accuracy on the test set."""
        _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return accuracy

    def predict(self, X):
        """Return a boolean prediction (threshold 0.5)."""
        value = self.model(X, training = False).numpy()[0][0]
        return value > 0.5

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def save(self, filepath='bg_model.h5'):
        """Persist model to disk."""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load a saved model from *filepath* and return an ``EMGModel`` instance."""
        if not os.path.exists(filepath):
            print(f"Error: Model '{filepath}' not found.")
            return None
        model = load_model(filepath)
        return cls(model=model, input_dim=model.input_shape[-1])

    # ------------------------------------------------------------------ #
    #  Data helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_and_preprocess_data(filepath='Data/Features/emg_final_4.csv'):
        """Load a feature CSV and return (X, y) numpy arrays.

        The feature matrix follows the CSV column order and uses every
        column except ``Output`` as an input feature.
        """
        import pandas as pd

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None, None

        if 'Output' not in df.columns:
            print(f"Error: File '{filepath}' must contain an 'Output' column.")
            return None, None

        feature_columns = [column for column in df.columns if column != 'Output']
        if not feature_columns:
            print(f"Error: File '{filepath}' does not contain any feature columns.")
            return None, None

        X = df[feature_columns].to_numpy()
        y = df['Output'].values
        return X, y

    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=13):
        """Split arrays into train/test sets."""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def load_and_extract_stream_features(file_path, segment_size=10, label_map=None):
        """Load a raw EMG stream CSV, extract time-domain features per segment,
        apply the binary label map, and return a feature DataFrame."""
        import ast
        import pandas as pd

        if label_map is None:
            label_map = {2: 1, 4: 1, 1: 0, 3: 0, 5: 0, 6: 0, 7: 0}

        _features_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'Testing Different Approaches', 'Features Sets'
        )
        if _features_dir not in sys.path:
            sys.path.insert(0, _features_dir)
        from all_features_sfs import calculate_emg_features

        print(f"Loading data from {file_path} ...")
        df = pd.read_csv(file_path)
        print(f"  Raw rows: {df.shape[0]}")

        def parse_value(v):
            try:
                return ast.literal_eval(str(v))
            except (ValueError, SyntaxError):
                return (0.0, 0.0)

        parsed = df['value'].apply(parse_value)
        df['filtered'] = parsed.apply(lambda t: t[0])
        df['envelope'] = parsed.apply(lambda t: t[1])

        df['label'] = df['level_number'].map(label_map)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        all_features = []
        for (sid, lvl), grp in df.groupby(['session_id', 'level_number'], sort=False):
            filt_vals = grp['filtered'].values
            env_vals = grp['envelope'].values
            output_label = grp['label'].iloc[0]

            n_samples = len(filt_vals)
            for start in range(0, n_samples, segment_size):
                filt_seg = filt_vals[start:start + segment_size]
                env_seg = env_vals[start:start + segment_size]

                if len(filt_seg) < 5:
                    continue

                combined = {}
                for k, v in calculate_emg_features(filt_seg).items():
                    combined[f'filt_{k}'] = v
                for k, v in calculate_emg_features(env_seg).items():
                    combined[f'env_{k}'] = v
                combined['Output'] = output_label
                all_features.append(combined)

        feat_df = pd.DataFrame(all_features)
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df.fillna(0, inplace=True)

        print(f"  Extracted {len(feat_df)} segments  |  "
              f"Features per segment: {len(feat_df.columns) - 1}")
        print(f"  Class distribution: {dict(feat_df['Output'].value_counts())}")
        return feat_df

    @staticmethod
    def save_training_graphs(history, filename='training_graphs.png'):
        """Save accuracy & loss curves as a PNG."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Training graphs saved to {filename}")