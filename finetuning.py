"""Finetuning pipeline — loads raw CSV, extracts features, finetunes a saved model."""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE

from model import EMGModel

SELECTED_FEATURES = ['filt_AR_2', 'filt_AR_3', 'env_AR_3', 'env_WAMP']
SEGMENT_LENGTH = 20


def main(csv_path='finetuningData.csv', model_path='optimized_model_2.h5'):
    """End-to-end finetuning: load data → engineer features → finetune → save."""

    # 1. Load raw stream and extract full time-domain features
    print(f"1. Loading data and extracting features from {csv_path}...")
    features_df = EMGModel.load_and_extract_stream_features(csv_path, segment_size=SEGMENT_LENGTH)
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
