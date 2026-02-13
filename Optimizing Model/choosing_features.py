import numpy as np
import pandas as pd
import ast
import os
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set seeds for reproducibility
np.random.seed(13)
tf.random.set_seed(13)

def calculate_emg_features(signal, ar_order=4, threshold=0.01):
    """
    Calculates EMG features from a signal array.
    """
    x = np.array(signal)
    N = len(x)
    i_idx = np.arange(1, N + 1)
    
    # --- Base Features ---
    wl = np.sum(np.abs(np.diff(x)))
    aac = wl / N
    dasdv = np.sqrt(np.sum(np.diff(x)**2) / (N - 1))
    
    res = AutoReg(x, lags=ar_order).fit()
    ar_coeffs = res.params[1:] 
    
    # CC calculation (from feature_engineering.py)
    cc = np.zeros(ar_order)
    if len(ar_coeffs) > 0:
        cc[0] = -ar_coeffs[0]
        for p in range(2, ar_order + 1):
             sum_val = 0
             for l in range(1, p):
                 sum_val += (1 - l/p) * ar_coeffs[l-1] * cc[p-l-1]
             cc[p-1] = -ar_coeffs[p-1] - sum_val
    
    # IEMG, MAV variants, SSI, VAR, Moments, RMS, V-Order, LOG
    iemg = np.sum(np.abs(x))
    mav = (1/N) * np.sum(np.abs(x))
    
    w1 = np.where((i_idx >= 0.25*N) & (i_idx <= 0.75*N), 1.0, 0.5)
    mav1 = (1/N) * np.sum(w1 * np.abs(x))
    
    w2 = np.ones(N)
    w2[i_idx < 0.25*N] = 4 * i_idx[i_idx < 0.25*N] / N
    w2[i_idx > 0.75*N] = 4 * (i_idx[i_idx > 0.75*N] - N) / N
    mav2 = (1/N) * np.sum(w2 * np.abs(x))
    
    ssi = np.sum(x**2)
    var = (1/(N-1)) * np.sum(x**2)
    tm3 = np.abs((1/N) * np.sum(x**3))
    tm4 = (1/N) * np.sum(x**4)
    tm5 = np.abs((1/N) * np.sum(x**5))
    rms = np.sqrt((1/N) * np.sum(x**2))
    v_val = ((1/N) * np.sum(np.abs(x)**2))**(0.5)
    log_det = np.exp((1/N) * np.sum(np.log(np.abs(x) + 1e-9)))

    # --- New Features from Images ---
    # ZC: Zero Crossing
    zc = 0
    for j in range(N - 1):
        cond1 = (x[j] * x[j+1] < 0) 
        cond2 = np.abs(x[j] - x[j+1]) >= threshold
        if cond1 and cond2:
            zc += 1

    # MYOP: Myopulse Percentage Rate
    myop = (1/N) * np.sum(np.abs(x) >= threshold)

    # WAMP: Willison Amplitude
    wamp = np.sum(np.abs(np.diff(x)) >= threshold)

    # SSC: Slope Sign Change
    ssc = 0
    for j in range(1, N - 1):
        if ((x[j] - x[j-1]) * (x[j] - x[j+1])) >= threshold:
            ssc += 1

    # MAVSLP: MAV Slope
    mavslp = np.mean(np.diff(np.abs(x)))

    # MHW & MTW: Modified Hamming / Trapezoidal
    mhw = np.sum((x * np.hamming(N))**2)
    mtw = np.sum((x**2) * np.blackman(N)) 

    features = {
        "WL": wl, "AAC": aac, "DASDV": dasdv, "IEMG": iemg, "MAV": mav,
        "MAV1": mav1, "MAV2": mav2, "SSI": ssi, "VAR": var, "TM3": tm3,
        "TM4": tm4, "TM5": tm5, "RMS": rms, "V_Order": v_val, "LOG": log_det,
        "ZC": zc, "MYOP": myop, "WAMP": wamp, "SSC": ssc, 
        "MAVSLP": mavslp, "MHW": mhw, "MTW": mtw
    }
    
    # Flatten AR and CC
    for i, val in enumerate(ar_coeffs):
        features[f"AR_{i+1}"] = val
    for i, val in enumerate(cc):
        features[f"CC_{i+1}"] = val
        
    return features

def load_and_preprocess_all(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None, None

    all_features = []
    
    for index, row in df.iterrows():
        try:
            filtered_values = ast.literal_eval(row['filtered_values'])
        except (ValueError, SyntaxError):
            continue
        
        level_number = row['level_number']
        if level_number in [2, 4]:
            output_label = 1
        elif level_number in [1, 3]:
            output_label = 0
        else:
            continue

        segment_size = 50
        for i in range(0, len(filtered_values), segment_size):
            segment = filtered_values[i:i+segment_size]
            if len(segment) > 0:
                feats = calculate_emg_features(segment)
                feats['Output'] = output_label
                all_features.append(feats)

    features_df = pd.DataFrame(all_features)
    print(f"Extracted features for {len(features_df)} segments.")
    return features_df

def build_custom_model(input_dim):
    # Hyperparameters from request
    # Hidden Layers: 1
    # Neurons: 32
    # Dropout: 0.3146
    
    model = Sequential()
    
    # First Hidden Layer
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3146))
    
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_evaluate(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_custom_model(X_train.shape[1])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)
    
    # Suppress output for loop
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, callbacks=[early_stopping], validation_split=0.2)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

def sequential_forward_selection(df, target_col='Output'):
    # Get all feature columns (excluding target)
    feature_cols = [c for c in df.columns if c != target_col]
    
    best_features = []
    best_accuracy = 0.0
    
    # SFS Loop
    print("\nStarting Sequential Forward Selection...")
    
    while len(best_features) < len(feature_cols):
        remaining_features = [f for f in feature_cols if f not in best_features]
        
        current_step_best_feature = None
        current_step_best_acc = 0.0
        
        # Test adding each remaining feature
        for feature in remaining_features:
            current_subset = best_features + [feature]
            
            X = df[current_subset].values
            y = df[target_col].values
            
            acc = train_and_evaluate(X, y)
            
            # print(f"  Testing {current_subset} -> Acc: {acc:.4f}")
            
            if acc > current_step_best_acc:
                current_step_best_acc = acc
                current_step_best_feature = feature
        
        # Check if improvement
        if current_step_best_acc > best_accuracy:
            best_accuracy = current_step_best_acc
            best_features.append(current_step_best_feature)
            print(f"Step {len(best_features)}: Added '{current_step_best_feature}' | New Best Accuracy: {best_accuracy:.4f}")
        else:
            print(f"No improvement with remaining features (Max Acc in this step: {current_step_best_acc:.4f}). Stopping.")
            break
            
    return best_features, best_accuracy

if __name__ == "__main__":
    input_file = 'emg_streamed_cleaned.csv'
    
    # 1. Load and extract all features
    df = load_and_preprocess_all(input_file)
    
    if df is not None and not df.empty:
        # 2. Run Forward Feature Selection
        best_feats, best_acc = sequential_forward_selection(df)
        
        print("\n" + "="*30)
        print("RESULT")
        print("="*30)
        print(f"Best Accuracy: {best_acc * 100:.2f}%")
        print(f"Best Feature Set ({len(best_feats)}): {best_feats}")
