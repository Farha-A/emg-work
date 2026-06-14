import os
import numpy as np


class EMGModel:
    def __init__(self, model):
        self._model = model

    @classmethod
    def load(cls, path):
        try:
            from tensorflow import keras
        except ImportError:
            import keras
        try:
            model = keras.models.load_model(path)
            print(f"   Model loaded from {path}")
            return cls(model)
        except Exception as e:
            print(f"ERROR: Failed to load model from {path}: {e}")
            return None

    def finetune(self, X_train, y_train, epochs=5, batch_size=32):
        self._model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self, X_test, y_test):
        result = self._model.evaluate(X_test, y_test, verbose=0)
        return result[1] if hasattr(result, "__len__") else result

    def save(self, path):
        import tensorflow as tf
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(path, "wb") as f:
            f.write(tflite_model)
        print(f"   Model saved to {path}")
