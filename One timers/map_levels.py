import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Map level values in a CSV file.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", help="Path to save the modified CSV (overwrites input if not provided)", default=None)
    
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error reading {args.input_csv}: {e}")
        sys.exit(1)
        
    if 'level_number' not in df.columns:
        print(f"Error: 'level_number' column not found in {args.input_csv}")
        sys.exit(1)
        
    # Define the mapping
    level_mapping = {
        1: 7,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6
    }
    
    # Apply the mapping. 
    # Values not in the mapping drop to NaN with map alone, 
    # so we use a lambda or fillna to retain non-mapped values.
    df['level_number'] = df['level_number'].map(lambda x: level_mapping.get(x, x))
    
    output_path = args.output if args.output else args.input_csv
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Successfully mapped levels and saved to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
