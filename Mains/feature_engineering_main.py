import sys
import os
import pandas as pd
import ast

# Add parent directory to sys.path to allow imports from the main directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering import FeatureEngineer

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'Data/emg_streamed_cleaned_2.csv'
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    extracted_features = []
    for index, row in df.iterrows():
        try:
            filtered_values = ast.literal_eval(row['filtered_values'])
        except (ValueError, SyntaxError):
            print(f"Error parsing filtered_values at row {index}")
            continue

        try:
            envelope_values = ast.literal_eval(row['envelope_values'])
        except (ValueError, SyntaxError):
            print(f"Error parsing envelope_values at row {index}")
            continue

        label = FeatureEngineer.get_output_label(row['level_number'])
        if label == -1:
            print(f"Warning: Unexpected level_number {row['level_number']} at row {index}")

        segment_size = 50
        max_len = max(len(filtered_values), len(envelope_values))
        for i in range(0, max_len, segment_size):
            filtered_segment = filtered_values[i : i + segment_size]
            envelope_segment = envelope_values[i : i + segment_size]
            if len(filtered_segment) > 0 or len(envelope_segment) > 0:
                filtered_feats = (
                    FeatureEngineer.calculate_emg_features(filtered_segment)
                    if len(filtered_segment) > 0
                    else {"AAC": 0.0, "WL": 0.0, "AR_2": 0.0}
                )
                envelope_feats = (
                    FeatureEngineer.calculate_emg_features(envelope_segment)
                    if len(envelope_segment) > 0
                    else {"AAC": 0.0, "WL": 0.0, "AR_2": 0.0}
                )
                extracted_features.append({
                    "filt_AAC": filtered_feats["AAC"],
                    "env_WL": envelope_feats["WL"],
                    "filt_AR_2": filtered_feats["AR_2"],
                    "Output": label,
                })

    features_df = pd.DataFrame(extracted_features)
    if not features_df.empty:
        features_df = features_df[['filt_AAC', 'env_WL', 'filt_AR_2', 'Output']]
        features_df = FeatureEngineer.apply_borderline_smote(features_df)

        print(features_df.head())
        features_df.to_csv('emg_features.csv', index=False)
        print("Features saved to emg_features.csv")
    else:
        print("No features were extracted.")
