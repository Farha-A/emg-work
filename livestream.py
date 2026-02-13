import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
import time
import random

# Import local modules
from feature_engineering import calculate_emg_features
from model import predict
from emg import *

def process_livestream(data_stream):
    """
    Simulates processing a live stream of data.
    
    Args:
        data_stream: An iterable that yields data points (numbers).
    """
    
    # Load the model
    # Note: Ensure bg_model.h5 exists. 
    try:
        model = load_model('bg_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize a queue with a maximum length of 50
    buffer = collections.deque(maxlen=50)
    
    print("Starting livestream processing...")
    
    for value in data_stream:
        # Add new value to the buffer
        buffer.append(value)
        
        # Check if we have enough data
        if len(buffer) == 50:
            # Convert buffer to list/array for feature calculation
            segment = list(buffer)
            
            # Calculate features
            # calculate_emg_features returns a dict with 'WL', 'AAC', 'DASDV', 'AR_Coeffs', 'Cepstral_Coeffs'
            features = calculate_emg_features(segment)
            
            # Flatten features to match model input (1, 11)
            # Order: WL, AAC, DASDV, AR (4), CC (4)
            
            wl = features['WL']
            aac = features['AAC']
            dasdv = features['DASDV']
            ar = features['AR_Coeffs']
            cc = features['Cepstral_Coeffs']
            
            # Ensure AR and CC are arrays/lists of length 4
            # (calculate_emg_features handles this, but good to be safe if dynamic)
            
            # Construct the feature vector
            feature_vector = np.concatenate(([wl, aac, dasdv], ar, cc))
            
            # Reshape for model input: (1, 11)
            input_data = feature_vector.reshape(1, -1)
            
            # Predict
            # Using the imported predict wrapper which handles the thresholding
            prediction = predict(model, input_data)
            
            print(f"Input: {value:.2f} | Buffer Full | Prediction: {prediction}")
        else:
            print(f"Input: {value:.2f} | Buffer Filling: {len(buffer)}/50")
            
        # Processing is driven by the generator's speed
        # time.sleep(0.05) # Removed to avoid double waiting

if __name__ == "__main__":
    from logger import Logger
    from constants import LOGGING_INTERVAL

    emg = EMGReader()
    # Give it a moment to connect
    time.sleep(2) 
    
    # Setup logger
    # Assuming current directory or a 'logs' directory
    # The logger uses the path to determine where to put emg_stream.csv. 
    # ensuring 'logs' dir exists might be good, or just use current dir.
    logger = Logger("livestream_data") 
    
    print("Starting Live Stream...")
    try:
        # Create the generator
        # We need a callable for the value. emg.envelope is a property/variable.
        # We need to capture the current value at the time of call.
        get_val = lambda: emg.envelope
        
        stream = logger.live_stream_generator(
            session_id="live_session",
            level_number=1,
            get_value_callable=get_val,
            interval=LOGGING_INTERVAL
        )
        
        process_livestream(stream)
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        emg.stop()
