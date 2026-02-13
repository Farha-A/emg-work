import pandas as pd
import ast
import os

def average_segments(lst, n_segments=3):
    n = len(lst) // n_segments
    if n == 0:
        return []
    parts = [lst[i*n : (i+1)*n] for i in range(n_segments)]
    return [sum(float(x) for x in group) / n_segments for group in zip(*parts)]

# Read the CSV file
file = pd.read_csv('emg_stream_h_2.csv')

# Parse the 'value' column from string to tuple using ast.literal_eval
# The string format is assumed to be "(val1, val2)"
file['parsed_value'] = file['value'].apply(ast.literal_eval)

# Split the tuple into two separate columns
file['filtered_value'] = file['parsed_value'].apply(lambda x: x[0])
file['envelope_value'] = file['parsed_value'].apply(lambda x: x[1])

# Group by session_id and level_number, and aggregate the new columns into lists
data = file.groupby(['session_id', 'level_number'])[['filtered_value', 'envelope_value']].agg(list).reset_index()

# Rename columns to match desired output
data.columns = ['session_id', 'level_number', 'filtered_values', 'envelope_values']

# Apply the averaging function to the lists
data['filtered_values'] = data.apply(lambda row: average_segments(row['filtered_values'], 2 if str(row['session_id']) == '595' else 3), axis=1)
data['envelope_values'] = data.apply(lambda row: average_segments(row['envelope_values'], 2 if str(row['session_id']) == '595' else 3), axis=1)

# Save to CSV
output_file = 'emg_streamed_cleaned.csv'
if os.path.exists(output_file):
    data.to_csv(output_file, mode='a', header=False, index=False)
else:
    data.to_csv(output_file, index=False)