import sys
import os

# Suppress TensorFlow logging to keep output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_h5_model.py <path_to_model.h5>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)
        
    try:
        print(f"Loading model from {model_path}...\n")
        model = tf.keras.models.load_model(model_path)
        
        print("=== Model Summary ===")
        model.summary()
        
        print("\n=== Detailed Parameters (Weights and Biases) ===")
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if not weights:
                print(f"\nLayer {i}: {layer.name} (No trainable parameters)")
                continue
                
            print(f"\nLayer {i}: {layer.name}")
            for j, w in enumerate(weights):
                param_type = "Weights" if j == 0 else ("Biases" if j == 1 else f"Param {j}")
                print(f"  {param_type} Shape: {w.shape}")
                print(f"  {param_type} Values:\n{w}\n")
                
    except Exception as e:
        print(f"Error loading model or reading parameters: {e}")

if __name__ == "__main__":
    main()
