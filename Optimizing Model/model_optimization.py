import numpy as np
import sys
import os

# Add the parent directory to sys.path so we can import model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model
import tensorflow as tf

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fitness_function(position, X_train, y_train, X_test, y_test):
    # Decode position to hyperparameters
    # position[0]: Hidden layers (1 or 2)
    # position[1]: Neuron count index (0-3 -> 8, 16, 32, 64)
    # position[2]: Dropout (0-0.5)
    
    hidden_layers = int(round(position[0]))
    neuron_index = int(round(position[1]))
    dropout = position[2]
    
    neuron_options = [8, 16, 32, 64]
    # Clamp just in case index goes out of bounds
    neuron_index = max(0, min(neuron_index, len(neuron_options) - 1))
    neurons = neuron_options[neuron_index]
    
    # Ensure hidden_layers is 1 or 2
    hidden_layers = max(1, min(hidden_layers, 2))
    
    # Ensure dropout is 0-0.5
    dropout = max(0.0, min(dropout, 0.5))
    
    print(f"Evaluating: Layers={hidden_layers}, Neurons={neurons}, Dropout={dropout:.4f}")
    
    # Build and Train Model
    # Since we are optimizing, we might want fewer epochs for speed, or full training?
    # User didn't specify, so let's stick to default or small number for speed during optimization.
    # But for accuracy, we need reasonable training. Let's use early stopping as defined in model.py
    # model.train_neural_network uses 5 epochs by default in the function signature, 
    # but the main block used 50. Let's use 20 to be faster but effective.
    
    try:
        if X_train is None or y_train is None:
             print("Error: Training data is None")
             return 1.0 # Worst fitness if data fails
             
        input_dim = X_train.shape[1]
        emg_model = model.EMGModel(input_dim=input_dim)
        emg_model.build(
            hidden_layers=hidden_layers,
            neurons=neurons,
            dropout=dropout,
            verbose=False
        )
        
        # Train
        emg_model.train(X_train, y_train, epochs=20, batch_size=32)
        
        # Evaluate
        accuracy = emg_model.evaluate(X_test, y_test)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Return minimization objective (1 - accuracy)
        return 1.0 - accuracy
        
    except Exception as e:
        print(f"Error in training: {e}")
        return 1.0 # Return worst fitness on error

def gwo_optimization(X_train, y_train, X_test, y_test, num_wolves=5, max_iter=10):
    # Dimensions: Layers, Neurons(Index), Dropout
    dim = 3
    
    # Bounds for each dimension
    # Layers: [1, 2]
    # Neurons Index: [0, 3]
    # Dropout: [0, 0.5]
    lb = np.array([1, 0, 0])
    ub = np.array([2, 3, 0.5])
    
    # Initialize implementation
    positions = np.zeros((num_wolves, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], num_wolves)
        
    # Alpha, Beta, Delta wolves
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    
    delta_pos = np.zeros(dim)
    delta_score = float("inf")
    
    print("Starting Grey Wolf Optimization...")
    
    for l in range(max_iter):
        print(f"\n--- Iteration {l+1}/{max_iter} ---")
        
        # Calculate fitness for each wolf
        for i in range(num_wolves):
            # Boundary handling
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            
            fitness = fitness_function(positions[i, :], X_train, y_train, X_test, y_test)
            
            # Update Alpha, Beta, Delta
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()
                
        print(f"Best Fitness (1-Acc): {alpha_score:.4f}")
        
        # Update positions
        a = 2 - l * (2 / max_iter) # a decreases linearly from 2 to 0
        
        for i in range(num_wolves):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                r1 = np.random.random()
                r2 = np.random.random()
                
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                r1 = np.random.random()
                r2 = np.random.random()
                
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                positions[i, j] = (X1 + X2 + X3) / 3
                
    return alpha_pos, alpha_score

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X, y = model.EMGModel.load_and_preprocess_data('C:\\University\\Grad!!!!!!!!!\\Data collection\\Cleaning\\Data\\Features\\emg_features_cc_2.csv')
    
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = model.EMGModel.split_data(X, y)
        print(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
        
        best_pos, best_score = gwo_optimization(X_train, y_train, X_test, y_test, num_wolves=30, max_iter=10)
        
        # Decode best solution
        hidden_layers = int(round(best_pos[0]))
        neuron_index = int(round(best_pos[1]))
        dropout = best_pos[2]
        neurons = [8, 16, 32, 64][max(0, min(neuron_index, 3))]
        
        with open("hyperparameters.txt", "w") as f:
            f.write("Optimization Complete!\n")
            f.write(f"Best Accuracy: {(1 - best_score) * 100:.2f}%\n")
            f.write("Best Hyperparameters:\n")
            f.write(f"  Hidden Layers: {hidden_layers}\n")
            f.write(f"  Neurons: {neurons}\n")
            f.write(f"  Dropout: {dropout:.4f}\n")
        
        # Train final model with best parameters
        print("\nTraining final model with best parameters...")
        final_emg_model = model.EMGModel(input_dim=X.shape[1])
        final_emg_model.build(
            hidden_layers=hidden_layers,
            neurons=neurons,
            dropout=dropout,
            verbose=True
        )
        final_emg_model.train(X_train, y_train, epochs=50, batch_size=32)
        final_acc = final_emg_model.evaluate(X_test, y_test)
        print(f"Final Model Test Accuracy: {final_acc * 100:.2f}%")
        
        final_emg_model.save('optimized_model.h5')
    else:
        print("Failed to load data.")
