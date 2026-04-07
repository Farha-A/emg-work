import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import collections
import time
import numpy as np

from config import EMG_BAUD, EMG_INPUT_MODE, EMG_PORT, SIMULATION_DURATION
from feature_engineering import FeatureEngineer
from model import EMGModel
from emg import *


def process_livestream(data_stream):
    """Process a live stream of EMG data and predict in real-time."""

    # Load the model
    try:
        emg_model = EMGModel.load('Models\\best_model.h5')
        if emg_model is None:
            return
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Sliding buffer of 50 samples
    buffer = collections.deque(maxlen=50)

    print("Starting livestream processing...")

    for value in data_stream:
        buffer.append(value)

        if len(buffer) == 50:
            segment = list(buffer)
            features = FeatureEngineer.calculate_emg_features(segment)
            feature_vector = np.array([[features['DASDV'], features['MYOP']]])
            prediction = emg_model.predict(feature_vector)
            print(f"Input: {value:.2f} | Buffer Full | Prediction: {prediction}")

            # Clear buffer when model predicts True 
            if prediction:
                print("Click detected! Clearing buffer.")
                buffer.clear()
        else:
            print(f"Input: {value:.2f} | Buffer Filling: {len(buffer)}/50")


if __name__ == "__main__":
    from logger import Logger
    from constants import LOGGING_INTERVAL

    use_keyboard_simulation = EMG_INPUT_MODE == "KEYBOARD"
    emg = EMGReader(
        port=EMG_PORT,
        baud=EMG_BAUD,
        simulate_with_space=use_keyboard_simulation,
        simulation_duration=SIMULATION_DURATION,
    )
    time.sleep(2)

    logger = Logger("livestream_data")

    print(
        f"Starting Live Stream... mode={EMG_INPUT_MODE} "
        f"source={'Space key' if use_keyboard_simulation else EMG_PORT}"
    )
    try:
        get_val = lambda: emg.envelope

        stream = logger.live_stream_generator(
            session_id="live_session",
            level_number=1,
            get_value_callable=get_val,
            interval=LOGGING_INTERVAL,
        )

        process_livestream(stream)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        emg.stop()
