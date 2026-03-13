import sys
import os

# Add parent directory to sys.path to allow imports from the main directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering import FeatureEngineer

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'Data/emg_stream.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'Data/emg_cleaned.csv'

    print(f"Loading raw CSV from: {input_file}")
    cleaned_df = FeatureEngineer.load_raw_csv(input_file)

    if cleaned_df.empty:
        print("Error: No data was loaded. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(cleaned_df)} rows.")
    print(cleaned_df.head())

    file_exists = os.path.exists(output_file)
    cleaned_df.to_csv(output_file, mode='a', index=False, header=not file_exists)
    print(f"Cleaned data {'appended to' if file_exists else 'saved to'}: {output_file}")
