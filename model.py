import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys

# input_dim=11 based on: 3 scalars (WL, AAC, DASDV) + 4 (AR) + 4 (CC)
INPUT_DIM = 11

def load_and_preprocess_data(filepath='emg_features.csv'):
    """
    Loads data from CSV, parses array columns, and prepares X and y.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None, None

    # Helper to parse string arrays like '[0.1 0.2 ...]'
    def parse_array_string(s):
        try:
            # Remove brackets and newlines, then split by whitespace
            clean_s = s.replace('[', '').replace(']', '').replace('\n', '')
            return np.fromstring(clean_s, sep=' ')
        except Exception:
            return np.zeros(4) # Fallback if parsing fails

    # Parse AR and CC columns
    # Assuming AR and CC are length 4 based on previous observation
    ar_data = np.stack(df['AR'].apply(parse_array_string).values)
    cc_data = np.stack(df['CC'].apply(parse_array_string).values)
    
    # Extract scalar features
    scalar_data = df[['WL', 'AAC', 'DASDV']].values
    
    # Combine all features into X
    X = np.hstack((scalar_data, ar_data, cc_data))
    
    # Extract labels
    y = df['Output'].values
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=13):
    """
    Splits data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_model(input_dim=INPUT_DIM, hidden_layers=2, neurons=16, dropout=0.0, verbose=False):
    # Build a neural network with variable structure
    model = Sequential()
    
    # Layer 1: First Hidden layer
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
        
    # Optional Layer 2: Additional Hidden layer
    if hidden_layers > 1:
        model.add(Dense(neurons, activation='relu'))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
            
    # Output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if verbose:
        # Display model summary
        model.summary()

    return model

def train_neural_network(model, X_train, y_train, epochs=5, batch_size=32):
    """
    Trains the Keras model.
    """
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping], validation_split=0.2)
    return model

def predict(model, X):
    value = model.predict(X)[0][0]
    return (value > 0.5)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and returns accuracy.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

def save_model_to_disk(model, filepath='bg_model.h5'):
    """
    Saves the trained model to disk.
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_and_predict(model_path, input_data):
    """
    Loads a saved model and makes predictions on input data.
    Input data should be preprocessed and match the model's input shape.
    """
    try:
        model = load_model(model_path)
        predictions = predict(model, input_data)
        return predictions
    except Exception as e:
        print(f"Error loading model or predicting: {e}")
        return None

if __name__ == "__main__":
    # Use command line argument for filename if provided, else default
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'emg_features.csv'
    
    print(f"Loading data from {file_path}...")
    X, y = load_and_preprocess_data(file_path)
    
    if X is not None and y is not None:
        print(f"Data loaded. Shape: X={X.shape}, y={y.shape}")
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        print(f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        print("Building model...")
        model = build_model(input_dim=X.shape[1], verbose=True)
        
        print("Training model...")
        train_neural_network(model, X_train, y_train, 50)
        
        print("Evaluating model...")
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
        
        save_model_to_disk(model)
