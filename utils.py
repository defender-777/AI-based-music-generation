import os

def save_model(model, path="model.keras"):
    model.save(path)

def load_model(path="model.keras"):
    from tensorflow.keras.models import load_model
    return load_model(path)
