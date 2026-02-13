import pandas as pd
import ast
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split

# Define average_segments helper (adapted from dataframe.py)
def average_segments(lst, n_segments=3):
    n = len(lst) // n_segments
    if n == 0:
        return []
    parts = [lst[i*n : (i+1)*n] for i in range(n_segments)]
    return [sum(float(x) for x in group) / n_segments for group in zip(*parts)]

# Define calculate_emg_features helper (from feature_engineering.py)
def calculate_emg_features(signal, ar_order=4):
    x = np.array(signal)
    N = len(x)
    
    # WL
    wl = np.sum(np.abs(np.diff(x)))
    # AAC
    aac = wl / N
    # DASDV
    dasdv = np.sqrt(np.sum(np.diff(x)**2) / (N - 1))
    
    # AR
    try:
        res = AutoReg(x, lags=ar_order).fit()
        ar_coeffs = res.params[1:] 
    except Exception:
        ar_coeffs = np.zeros(ar_order)

    # CC
    cc = np.zeros(ar_order)
    if len(ar_coeffs) > 0:
        cc[0] = -ar_coeffs[0] 
        for p in range(2, ar_order + 1):
            sum_val = 0
            for l in range(1, p):
                sum_val += (1 - l/p) * ar_coeffs[l-1] * cc[p-l-1]
            cc[p-1] = -ar_coeffs[p-1] - sum_val

    return {
        "WL": wl,
        "AAC": aac,
        "DASDV": dasdv,
        "AR_Coeffs": ar_coeffs,
        "Cepstral_Coeffs": cc
    }

# 1.1 & 1.2
def load_data(csv_path='finetuningData.csv'):
    """
    Loads CSV, processes it (excluding special session 595 logic), and returns a DataFrame.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return pd.DataFrame()

    file = pd.read_csv(csv_path)
    
    if 'value' not in file.columns:
        print("Error: 'value' column missing.")
        return pd.DataFrame()

    try:
        file['parsed_value'] = file['value'].apply(ast.literal_eval)
    except Exception as e:
        print(f"Error parsing values: {e}")
        return pd.DataFrame()

    file['filtered_value'] = file['parsed_value'].apply(lambda x: x[0])
    file['envelope_value'] = file['parsed_value'].apply(lambda x: x[1])

    data = file.groupby(['session_id', 'level_number'])[['filtered_value', 'envelope_value']].agg(list).reset_index()
    data.columns = ['session_id', 'level_number', 'filtered_values', 'envelope_values']

    # Apply average_segments with n_segments=3 for ALL sessions (excluding 595 check)
    data['filtered_values'] = data.apply(lambda row: average_segments(row['filtered_values'], 3), axis=1)
    data['envelope_values'] = data.apply(lambda row: average_segments(row['envelope_values'], 3), axis=1)

    return data

# 2.1 & 2.2
def engineer_features(df):
    """
    Engineers features from the processed DataFrame.
    """
    if df.empty:
        return pd.DataFrame()
        
    extracted_features = []
    
    for _, row in df.iterrows():
        filtered_values = row['filtered_values']
        level_number = row['level_number']
        
        # Determine Output
        if level_number in [2, 4]:
            output_label = 1
        elif level_number in [1, 3]:
            output_label = 0
        else:
            output_label = -1
            
        segment_size = 50
        # Iterate through filtered_values
        for i in range(0, len(filtered_values), segment_size):
            segment = filtered_values[i:i+segment_size]
            if len(segment) > 0:
                features = calculate_emg_features(segment)
                feature_row = {
                    "WL": features["WL"],
                    "AAC": features["AAC"],
                    "DASDV": features["DASDV"],
                    "AR": features["AR_Coeffs"],
                    "CC": features["Cepstral_Coeffs"],
                    "Output": output_label
                }
                extracted_features.append(feature_row)
                
    features_df = pd.DataFrame(extracted_features)
    if not features_df.empty:
        features_df = features_df[['WL', 'AAC', 'DASDV', 'AR', 'CC', 'Output']]
        
    return features_df

# 3.1 & 3.2
def finetune_model(model_path, df):
    """
    Loads a model and fine-tunes it on the provided DataFrame.
    """
    if df.empty:
        print("No data to finetune on.")
        return None
        
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_path}' not found.")
        return None

    # Preprocess data (similar to model.py)
    # Parse AR/CC if they are strings (they shouldn't be if coming from engineer_features, 
    # but for robustness we check or assume list/array)
    
    # If engineer_features returns lists/arrays, we need to stack them
    try:
        ar_data = np.stack(df['AR'].values)
        cc_data = np.stack(df['CC'].values)
        scalar_data = df[['WL', 'AAC', 'DASDV']].values
        
        X = np.hstack((scalar_data, ar_data, cc_data))
        y = df['Output'].values
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    
    # Load model
    model = load_model(model_path)
    
    # Train
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, callbacks=[early_stopping], validation_split=0.2)
    
    # Evaluate (optional but good for log)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Finetuned Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# 4.1
def save_model(model):
    """
    Saves the model to 'Models/finetuned_model.keras'.
    """
    if model is None:
        print("No model to save.")
        return

    output_dir = 'Models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'finetuned_model.h5')
    model.save(output_path)
    print(f"Model saved to {output_path}")

# 5.1
def main(csv_path='finetuningData.csv', model_path='bg_model.h5'):
    """
    Main execution pipeline.
    """
    print(f"1. Loading data from {csv_path}...")
    df_raw = load_data(csv_path)
    
    print("2. Engineering features...")
    df_features = engineer_features(df_raw)
    
    print(f"3. Finetuning model from {model_path}...")
    model = finetune_model(model_path, df_features)
    
    print("4. Saving model...")
    save_model(model)
    
if __name__ == "__main__":
    # Example usage
    # Ensure a model file and csv file exist before running direct
    main()
