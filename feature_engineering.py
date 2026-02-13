import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def calculate_emg_features(signal, ar_order=4):
    """
    Calculates EMG features from a signal array based on the provided images.
    
    Parameters:
    signal (np.array): The input EMG time-series data.
    ar_order (int): The order P for Auto-regressive and Cepstral coefficients.
    
    Returns:
    dict: A dictionary containing the calculated features.
    """
    x = np.array(signal)
    N = len(x)
    
    # 2.1.11 Waveform Length (WL)
    # Sum of absolute differences between consecutive samples
    wl = np.sum(np.abs(np.diff(x)))
    
    # 2.1.12 Average Amplitude Change (AAC)
    # WL divided by the number of samples
    aac = wl / N
    
    # 2.1.13 Difference Absolute Standard Deviation Value (DASDV)
    # Square root of the mean of squared differences
    dasdv = np.sqrt(np.sum(np.diff(x)**2) / (N - 1))
    
    # 2.1.23 Auto-regressive Coefficients (AR)
    # Using Burg's method or Yule-Walker to estimate coefficients a_p
    # Note: Using a simplified linalg approach for demonstration
    res = AutoReg(x, lags=ar_order).fit()
    # statsmodels returns [intercept, a1, a2, ...], we take the coefficients a_p
    # Note: Signs may vary by convention; standard AR is xi = sum(ap * xi-p)
    ar_coeffs = res.params[1:] 
    
    # 2.1.24 Cepstral Coefficients (CC)
    # Derived recursively from AR coefficients (ap)
    cc = np.zeros(ar_order)
    if len(ar_coeffs) > 0:
        cc[0] = -ar_coeffs[0] # c1 = -a1
        for p in range(2, ar_order + 1):
            sum_val = 0
            for l in range(1, p):
                sum_val += (1 - l/p) * ar_coeffs[l-1] * cc[p-l-1]
            cc[p-1] = -ar_coeffs[p-1] - sum_val

    return {
        "WL": wl,
        "AAC": aac,
        "DASDV": dasdv,
        "AR_Coeffs": ar_coeffs,
        "Cepstral_Coeffs": cc
    }


if __name__ == "__main__":
    import pandas as pd
    import ast

    # Load the cleaned data
    input_file = 'emg_streamed_cleaned.csv'
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        exit()

    extracted_features = []

    # Iterate through each row in the dataframe
    for index, row in df.iterrows():
        # Parse the string representation of the list
        try:
            filtered_values = ast.literal_eval(row['filtered_values'])
        except (ValueError, SyntaxError):
            print(f"Error parsing filtered_values at row {index}")
            continue
        
        level_number = row['level_number']
        
        # Determine Output based on level_number
        if level_number in [2, 4]:
            output_label = 1
        elif level_number in [1, 3]:
            output_label = 0
        else:
            output_label = -1 # Or some other default/error value
            print(f"Warning: Unexpected level_number {level_number} at row {index}")

        # Chunk the data into 50-value segments
        segment_size = 50
        for i in range(0, len(filtered_values), segment_size):
            segment = filtered_values[i:i+segment_size]
            
            # Ensure we have enough data for a segment 
            # assuming standard chunking.
            if len(segment) > 0:
                features = calculate_emg_features(segment)
                
                # Construct the row for the new DataFrame
                feature_row = {
                    "WL": features["WL"],
                    "AAC": features["AAC"],
                    "DASDV": features["DASDV"],
                    "AR": features["AR_Coeffs"], # Keeping as array/list
                    "CC": features["Cepstral_Coeffs"], # Keeping as array/list
                    "Output": output_label
                }
                extracted_features.append(feature_row)

    # Create the new DataFrame
    features_df = pd.DataFrame(extracted_features)
    
    # Reorder columns to match request: WL, AAC, DASDV, AR, CC, Output
    features_df = features_df[['WL', 'AAC', 'DASDV', 'AR', 'CC', 'Output']]

    print(features_df.head())
    
    # Save to CSV (Optional but good practice)
    features_df.to_csv('emg_wl.csv', index=False)
    print("Features saved to emg_wl.csv")