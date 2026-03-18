import numpy as np
from model import load_and_predict

# Dummy data: 1 sample with 3 features
# AAC (1) + WL (1) + AR_2 (1) = 3
input_data = np.random.random((1, 3))

print("Loading model and predicting...")
prediction = load_and_predict('bg_model.h5', input_data)

if prediction is not None:
    print(f"Prediction successful: {prediction}")
else:
    print("Prediction failed.")
