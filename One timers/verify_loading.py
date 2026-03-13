import numpy as np
from model import load_and_predict

# Dummy data: 1 sample with 2 features
# DASDV (1) + MYOP (1) = 2
input_data = np.random.random((1, 2))

print("Loading model and predicting...")
prediction = load_and_predict('bg_model.h5', input_data)

if prediction is not None:
    print(f"Prediction successful: {prediction}")
else:
    print("Prediction failed.")
