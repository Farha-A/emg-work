import numpy as np
import pandas as pd
import os
import sys
import datetime

# Suppress TensorFlow oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Set seeds for reproducibility
np.random.seed(13)
tf.random.set_seed(13)


def parse_hyperparameters(file_path):
    """
    Reads hyperparameters from a text file.
    Expected format:
        Hidden Layers: <int>
        Neurons: <int>
        Dropout: <float>
    """
    params = {'hidden_layers': 2, 'neurons': 8, 'dropout': 0.2}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if 'Hidden Layers:' in line:
                    params['hidden_layers'] = int(line.split(':')[-1].strip())
                elif 'Neurons:' in line:
                    params['neurons'] = int(line.split(':')[-1].strip())
                elif 'Dropout:' in line:
                    params['dropout'] = float(line.split(':')[-1].strip())
        print(f"Loaded hyperparameters: {params}")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, using defaults: {params}")
    return params


def build_custom_model(input_dim, hyperparams):
    """
    Builds a Keras model using hyperparameters loaded from file.
    """
    hidden_layers = hyperparams['hidden_layers']
    neurons = hyperparams['neurons']
    dropout = hyperparams['dropout']

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for i in range(hidden_layers):
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_and_evaluate_auc(X, y, hyperparams):
    """
    Trains the model and returns AUC-ROC and accuracy on the test set.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = build_custom_model(X_train_resampled.shape[1], hyperparams)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)

    # Suppress output for loop
    model.fit(X_train_resampled, y_train_resampled, epochs=20, batch_size=32, verbose=0, callbacks=[early_stopping], validation_split=0.2)

    # Get predicted probabilities for AUC-ROC
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    auc = roc_auc_score(y_test, y_pred_proba)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return auc, accuracy


def sequential_forward_selection(df, hyperparams, target_col='Output'):
    # Get all feature columns (excluding target)
    feature_cols = [c for c in df.columns if c != target_col]

    best_features = []
    best_auc = 0.0
    best_auc_associated_acc = 0.0
    best_combined_score = 0.0

    # SFS Loop
    print("\nStarting Sequential Forward Selection (AUC-ROC + Accuracy)...")

    while len(best_features) < len(feature_cols):
        remaining_features = [f for f in feature_cols if f not in best_features]

        current_step_best_feature = None
        current_step_best_auc = 0.0
        current_step_best_acc = 0.0
        current_step_best_score = 0.0

        # Test adding each remaining feature
        for feature in remaining_features:
            current_subset = best_features + [feature]

            X = df[current_subset].values
            y = df[target_col].values

            auc, acc = train_and_evaluate_auc(X, y, hyperparams)
            
            # Use sum of AUC and Accuracy to find the best balanced feature set
            combined_score = auc + acc

            if combined_score > current_step_best_score:
                current_step_best_score = combined_score
                current_step_best_auc = auc
                current_step_best_feature = feature
                current_step_best_acc = acc

        # Check if improvement
        if current_step_best_score > best_combined_score:
            best_combined_score = current_step_best_score
            best_auc = current_step_best_auc
            best_auc_associated_acc = current_step_best_acc
            best_features.append(current_step_best_feature)
            print(f"Step {len(best_features)}: Added '{current_step_best_feature}' | New Best AUC-ROC: {best_auc:.4f} | Accuracy: {best_auc_associated_acc:.4f} | Combined: {best_combined_score:.4f}")
        else:
            print(f"No improvement with remaining features (Max Combined Score in this step: {current_step_best_score:.4f}). Stopping.")
            break

    return best_features, best_auc, best_auc_associated_acc


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        # Use 'a' to append instead of 'w' to overwrite
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, 'choosing_features_auc_output.txt')
    sys.stdout = Logger(log_path)

    # Add a visual separator for new runs in the log
    print(f"\n{'='*50}")
    print(f"Starting new run at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    features_file = r'C:\University\Grad!!!!!!!!!\Data collection\Cleaning\Data\Features\features_no_corr_no_zv.csv'
    hyperparams_file = os.path.join(script_dir, 'hyperparameters.txt')

    # 1. Load hyperparameters from file
    hyperparams = parse_hyperparameters(hyperparams_file)

    # 2. Load pre-computed features
    print(f"Loading features from {features_file}...")
    df = pd.read_csv(features_file)
    print(f"Loaded {len(df)} rows with {len(df.columns) - 1} features.")

    if df is not None and not df.empty:
        # 3. Run Forward Feature Selection using AUC-ROC
        best_feats, best_auc, best_acc = sequential_forward_selection(df, hyperparams)

        print("\n" + "="*30)
        print("RESULT")
        print("="*30)
        print(f"Best Combined Score: {(best_auc + best_acc):.4f}")
        print(f"Best AUC-ROC: {best_auc:.4f}")
        print(f"Associated Accuracy: {best_acc:.4f}")
        print(f"Best Feature Set ({len(best_feats)}): {best_feats}")
