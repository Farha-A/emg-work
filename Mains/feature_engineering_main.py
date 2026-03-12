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
                import numpy as np
                filtered_cc = (
                    FeatureEngineer.calculate_emg_features(filtered_segment)["Cepstral_Coeffs"]
                    if len(filtered_segment) > 0
                    else np.zeros(4)
                )
                envelope_cc = (
                    FeatureEngineer.calculate_emg_features(envelope_segment)["Cepstral_Coeffs"]
                    if len(envelope_segment) > 0
                    else np.zeros(4)
                )
                extracted_features.append({
                    "Filtered_CC": filtered_cc,
                    "Envelope_CC": envelope_cc,
                    "Output": label,
                })

    features_df = pd.DataFrame(extracted_features)
    if not features_df.empty:
        features_df = features_df[['Filtered_CC', 'Envelope_CC', 'Output']]

        print(features_df.head())
        features_df.to_csv('emg_features_cc.csv', index=False)
        print("Features saved to emg_features_cc.csv")
    else:
        print("No features were extracted.")
