import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os

def parse_array_string(s):
    if pd.isna(s):
        return np.zeros(4)
    # Remove brackets and newline characters
    s = s.replace('[', '').replace(']', '').replace('\n', ' ')
    # Parse floats separated by whitespace
    return np.fromstring(s, sep=' ')

def main():
    csv_path = 'emg_features.csv'
    model_path = os.path.join('Models', 'optimized_model.h5')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    
    print(f"1. Loading engineered features from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("2. Preprocessing arrays...")
    df['AR'] = df['AR'].apply(parse_array_string)
    df['CC'] = df['CC'].apply(parse_array_string)
    
    ar_data = np.stack(df['AR'].values)
    cc_data = np.stack(df['CC'].values)
    scalar_data = df[['WL', 'AAC', 'DASDV']].values
    
    X = np.hstack((scalar_data, ar_data, cc_data))
    y = df['Output'].values
    
    # Filter out output_label = -1 (unexpected level numbers)
    valid_indices = y != -1
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"   Data shape after filtering: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    
    print(f"3. Loading model from {model_path}...")
    model = load_model(model_path, compile=False)
    
    print("4. Compiling model with new optimizer...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    print("5. Finetuning model...")
    # Monitor val_accuracy for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train, 
        epochs=15, 
        batch_size=32, 
        verbose=1, 
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    print("5. Evaluating finetuned model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Finetuned Model Test Accuracy: {accuracy * 100:.2f}%")
    
    output_dir = 'Models'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'finetuned_model.h5')
    
    print(f"6. Saving new model to {output_path}...")
    model.save(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
