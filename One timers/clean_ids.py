import pandas as pd
import os

def clean_files():
    ids_to_remove = [292, 221]
    
    # Clean info_h.csv (no header)
    if os.path.exists('info_h_2.csv'):
        print("Cleaning info_h.csv...")
        # Read without header
        df_info = pd.read_csv('info_h_2.csv', header=None)
        # Filter rows where column 0 is not in ids_to_remove
        df_info_clean = df_info[~df_info[0].isin(ids_to_remove)]
        # Save back without header and index
        df_info_clean.to_csv('info_h_2.csv', index=False, header=False)
        print("info_h_2.csv cleaned.")

    # Clean emg_stream_h.csv (with header)
    if os.path.exists('emg_stream_h_2.csv'):
        print("Cleaning emg_stream_h.csv...")
        df_emg = pd.read_csv('emg_stream_h_2.csv')
        # Filter rows where session_id is not in ids_to_remove
        # Ensure session_id is treated as same type (int)
        df_emg_clean = df_emg[~df_emg['session_id'].isin(ids_to_remove)]
        df_emg_clean.to_csv('emg_stream_h_2.csv', index=False)
        print("emg_stream_h_2.csv cleaned.")

if __name__ == "__main__":
    clean_files()
