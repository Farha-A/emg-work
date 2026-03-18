import sys
import os
import argparse

# Add the parent 'Cleaning' directory (and 'Optimizing Model Performance') to sys.path
# so we can import load_and_preprocess_all from choosing_features.py
script_dir = os.path.dirname(os.path.abspath(__file__))
cleaning_dir = os.path.dirname(script_dir)
optimizing_dir = os.path.join(cleaning_dir, 'Optimizing Model Performance')

if optimizing_dir not in sys.path:
    sys.path.append(optimizing_dir)

try:
    from choosing_features import load_and_preprocess_all
except ImportError as e:
    print(f"Error importing choosing_features module: {e}")
    print(f"Tried looking in: {optimizing_dir}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract features from a CSV file combining choosing_features.py logic.")
    parser.add_argument("input_csv", type=str, help="Absolute or relative path to the input CSV file.")
    parser.add_argument("-o", "--output_csv", type=str, help="Path to save the output feature dataset CSV file. (Optional)")
    
    args = parser.parse_args()
    
    input_csv = args.input_csv
    if not os.path.exists(input_csv):
        print(f"Error: The input file '{input_csv}' does not exist.")
        sys.exit(1)
        
    print(f"Extracting features from: {input_csv}")
    
    # Use the function from choosing_features.py to process the CSV
    features_df = load_and_preprocess_all(input_csv)
    
    if features_df is not None and not features_df.empty:
        # If output destination is not provided, save alongside the input file
        if args.output_csv:
            output_csv = args.output_csv
        else:
            base_name = os.path.splitext(os.path.basename(input_csv))[0]
            output_dir = os.path.dirname(os.path.abspath(input_csv))
            output_csv = os.path.join(output_dir, f"{base_name}_features.csv")
            
        print(f"Saving extracted features dataset to: {output_csv}")
        # Save to CSV without the index column
        features_df.to_csv(output_csv, index=False)
        print("Dataset creation complete.")
    else:
        print("Failed to extract features. Dataset might be empty or invalid.")

if __name__ == "__main__":
    main()
