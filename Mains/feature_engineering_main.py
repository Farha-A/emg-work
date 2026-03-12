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

        label = FeatureEngineer.get_output_label(row['level_number'])
        if label == -1:
            print(f"Warning: Unexpected level_number {row['level_number']} at row {index}")

        segment_size = 50
        for i in range(0, len(filtered_values), segment_size):
            segment = filtered_values[i : i + segment_size]
            if len(segment) > 0:
                features = FeatureEngineer.calculate_emg_features(segment)
                extracted_features.append({
                    "CC": features["Cepstral_Coeffs"],
                    "Output": label,
                })

    features_df = pd.DataFrame(extracted_features)
    if not features_df.empty:
        features_df = features_df[['CC', 'Output']]

        print(features_df.head())
        features_df.to_csv('emg_features_cc.csv', index=False)
        print("Features saved to emg_features_cc.csv")
    else:
        print("No features were extracted.")
