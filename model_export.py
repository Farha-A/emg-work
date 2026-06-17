# Save this as d:\Projects\emg-work\One timers\export_model_to_json.py
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EMGModel

def export_to_json(h5_path, json_path):
    emg = EMGModel.load(h5_path)
    if not emg:
        print("Failed to load model.")
        return
        
    layers_data = []
    # Extract weights and biases from each Dense layer
    for layer in emg.model.layers:
        weights = layer.get_weights()
        if len(weights) == 2:
            w, b = weights
            layers_data.append({
                'name': layer.name,
                'weights': w.tolist(),
                'biases': b.tolist()
            })
            
    with open(json_path, 'w') as f:
        json.dump(layers_data, f)
        
    print(f"Model weights successfully exported to {json_path}")

if __name__ == "__main__":
    export_to_json('Models\\final_model.h5', 'Models\\final_model.json')
