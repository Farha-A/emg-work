"""Finetuning pipeline — loads raw CSV, extracts features, finetunes a saved model."""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from feature_engineering import FeatureEngineer
from model import EMGModel


def main(csv_path='finetuningData.csv', model_path='optimized_model_2.h5'):
    """End-to-end finetuning: load data → engineer features → finetune → save."""

    # 1. Load and clean raw CSV
    print(f"1. Loading data from {csv_path}...")
    data = FeatureEngineer.load_raw_csv(csv_path, n_segments=3)
    if data.empty:
        return

    # 2. Extract features
    print("2. Engineering features...")
    features_df = FeatureEngineer.extract_features_from_df(data)
    if features_df.empty:
        print("No features extracted.")
        return

    # 3. Prepare arrays
    filtered_dasdv = features_df['filt_DASDV'].values.reshape(-1, 1)
    filtered_myop = features_df['filt_MYOP'].values.reshape(-1, 1)
    X = np.hstack([filtered_dasdv, filtered_myop])
    y = features_df['Output'].values

    valid = y != -1
    X, y = X[valid], y[valid]
    print(f"   Data shape after filtering: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

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
