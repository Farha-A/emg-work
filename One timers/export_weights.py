import json
import sys
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

def export_h5_to_json(h5_path, json_path):
    print(f"Loading model from {h5_path}...")
    model = tf.keras.models.load_model(h5_path)
    
    model_data = []
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
            
        layer_data = {
            'name': layer.name,
            'weights': weights[0].tolist(),
            'biases': weights[1].tolist() if len(weights) > 1 else []
        }
        model_data.append(layer_data)
        
    with open(json_path, 'w') as f:
        json.dump(model_data, f)
        
    print(f"Model weights exported to {json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_weights.py <input.h5> <output.json>")
        sys.exit(1)
    export_h5_to_json(sys.argv[1], sys.argv[2])
