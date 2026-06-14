"""Finetuning pipeline — loads raw CSV, extracts features, finetunes a saved model."""

import os
import ast
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE

from model import EMGModel

SELECTED_FEATURES = ['filt_AR_2', 'filt_AR_3', 'env_AR_3', 'env_WAMP']
SEGMENT_LENGTH = 20
LABEL_MAP = {2: 1, 4: 1, 1: 0, 3: 0, 5: 0, 6: 0, 7: 0}


def calculate_emg_features(signal, ar_order=4, threshold=0.01):
    x = np.array(signal, dtype=float)
    N = len(x)
    if N == 0:
        return {}
    try:
        ar_coeffs = AutoReg(x, lags=ar_order).fit().params[1:]
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        ar_coeffs = np.zeros(ar_order)
    wamp = np.sum(np.abs(np.diff(x)) >= threshold)
    features = {"WAMP": wamp}
    for i, val in enumerate(ar_coeffs):
        features[f"AR_{i + 1}"] = val
    return features


def load_and_extract_features(csv_path, segment_size):
    df = pd.read_csv(csv_path)

    def parse_value(v):
        try:
            return ast.literal_eval(str(v))
        except (ValueError, SyntaxError):
            return (0.0, 0.0)

    parsed = df['value'].apply(parse_value)
    df['filtered'] = parsed.apply(lambda t: t[0])
    df['envelope'] = parsed.apply(lambda t: t[1])
    df['label'] = df['level_number'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    rows = []
    for (_, _), grp in df.groupby(['session_id', 'level_number'], sort=False):
        filt_vals = grp['filtered'].values
        env_vals = grp['envelope'].values
        output_label = grp['label'].iloc[0]
        for start in range(0, len(filt_vals), segment_size):
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
            rows.append(combined)

    feat_df = pd.DataFrame(rows)
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.fillna(0, inplace=True)
    return feat_df


def main(csv_path='finetuningData.csv', model_path='optimized_model_2.h5'):
    """End-to-end finetuning: load data → engineer features → finetune → save."""

    # 1. Load raw stream and extract features
    print(f"1. Loading data and extracting features from {csv_path}...")
    features_df = load_and_extract_features(csv_path, SEGMENT_LENGTH)
    if features_df is None or features_df.empty:
        print("No features extracted.")
        return

    # 2. Select the target features
    missing = [f for f in SELECTED_FEATURES if f not in features_df.columns]
    if missing:
        print(f"ERROR: missing features in data: {missing}")
        return

    X = features_df[SELECTED_FEATURES].values
    y = features_df['Output'].values

    valid = y != -1
    X, y = X[valid], y[valid]
    print(f"   Selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    print(f"   Data shape after filtering: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # 3. Balance training set
    smote = BorderlineSMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE – Train: {X_train.shape}")

    # 4. Load existing model and finetune
    print(f"3. Loading model from {model_path}...")
    emg_model = EMGModel.load(model_path)
    if emg_model is None:
        return

    print("4. Finetuning model...")
    emg_model.finetune(X_train, y_train, epochs=5, batch_size=32)

    # 5. Evaluate
    accuracy = emg_model.evaluate(X_test, y_test)
    print(f"   Finetuned Model Accuracy: {accuracy * 100:.2f}%")

    # 6. Save
    output_path = os.path.join('Models', 'finetuned_model.h5')
    emg_model.save(output_path)


if __name__ == "__main__":
    main()
