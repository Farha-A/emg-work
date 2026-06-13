import numpy as np
import json
import os

class NumpyEMGModel:
    """A pure NumPy implementation of the EMG model for inference.
    
    This avoids the heavy TensorFlow dependency for the compiled executable.
    """
    
    def __init__(self, json_path=None):
        self.layers = []
        if json_path:
            self.load(json_path)
            
    @classmethod
    def load(cls, filepath):
        """Load a saved model from *filepath* and return an instance."""
        if not os.path.exists(filepath):
            print(f"Error: Model '{filepath}' not found.")
            return None
        instance = cls()
        instance._load(filepath)
        return instance
            
    def _load(self, json_path):
        """Load weights and biases from a JSON file."""
        if not os.path.exists(json_path):
            print(f"Error: Model file '{json_path}' not found.")
            return False
            
        with open(json_path, 'r') as f:
            model_data = json.load(f)
            
        self.layers = []
        for layer_data in model_data:
            self.layers.append({
                'name': layer_data['name'],
                'weights': np.array(layer_data['weights']),
                'biases': np.array(layer_data['biases'])
            })
        print(f"Loaded NumPy model from {json_path}")
        return True
            
    def relu(self, x):
        return np.maximum(0, x)
        
    def sigmoid(self, x):
        # Clip to avoid overflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
        
    def predict(self, X):
        """Forward pass matching the Keras model: Dense (ReLU) -> ... -> Dense (Sigmoid)."""
        A = X
        for i, layer in enumerate(self.layers):
            W = layer['weights']
            b = layer['biases']
            
            Z = np.dot(A, W) + b
            
            # Assuming all layers except the last are ReLU, and the last is Sigmoid.
            # We can also check the layer name or just infer from position.
            if i == len(self.layers) - 1:
                A = self.sigmoid(Z)
            else:
                A = self.relu(Z)
                
        # Return a boolean prediction (threshold 0.5) matching the original logic
        value = A[0][0]
        return value > 0.5
