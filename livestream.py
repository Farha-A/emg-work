import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import collections
import time
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

from config import EMG_BAUD, EMG_INPUT_MODE, EMG_PORT, SIMULATION_DURATION
from model import EMGModel
from emg import *

SELECTED_FEATURES = ['filt_AR_2', 'filt_AR_3', 'env_AR_3', 'env_WAMP']


def calculate_emg_features(signal, ar_order=4, threshold=0.01):
    x = np.array(signal, dtype=float)
    N = len(x)
    if N == 0:
        return {}
    try:
        ar_coeffs = AutoReg(x, lags=ar_order).fit().params[1:]
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        ar_coeffs = np.zeros(ar_order)
    wamp = np.sum(np.abs(np.diff(x)) >= threshold)
    features = {"WAMP": wamp}
    for i, val in enumerate(ar_coeffs):
        features[f"AR_{i + 1}"] = val
    return features
SEGMENT_LENGTH = 20


def process_livestream(data_stream):
    """Process a live stream of EMG data and predict in real-time."""

    # Load the model
    try:
        emg_model = EMGModel.load('Models\\finetuned_model.tflite')
        if emg_model is None:
            return
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Sliding buffer of SEGMENT_LENGTH samples (each entry is a (filtered, envelope) tuple)
    buffer = collections.deque(maxlen=SEGMENT_LENGTH)

    print("Starting livestream processing...")

    for value in data_stream:
        buffer.append(value)

        if len(buffer) == SEGMENT_LENGTH:
            segment = list(buffer)

            # Split the (filtered, envelope) tuples into separate arrays
            filtered_seg = np.array([v[0] for v in segment])
            envelope_seg = np.array([v[1] for v in segment])

            # Extract full time-domain features from both signals
            filt_feats = calculate_emg_features(filtered_seg)
            env_feats = calculate_emg_features(envelope_seg)

            # Build the feature vector for the selected features
            combined = {}
            for k, v in filt_feats.items():
                combined[f'filt_{k}'] = v
            for k, v in env_feats.items():
                combined[f'env_{k}'] = v

            feature_vector = np.array([[combined[f] for f in SELECTED_FEATURES]])
            prediction = emg_model.predict(feature_vector)
            print(f"Input: {value} | Buffer Full | Prediction: {prediction}")

            # Clear buffer when model predicts True 
            if prediction:
                print("Click detected! Clearing buffer.")
                buffer.clear()
        else:
            print(f"Input: {value} | Buffer Filling: {len(buffer)}/{SEGMENT_LENGTH}")


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
        get_val = lambda: (emg.filtered, emg.envelope)

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
