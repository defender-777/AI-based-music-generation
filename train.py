import tensorflow as tf
import numpy as np

def create_dataset(notes, seq_length=25):
    inputs = []
    targets = []

    for i in range(len(notes["pitch"]) - seq_length):
        input_seq = {key: val[i:i+seq_length] for key, val in notes.items()}
        target_seq = {key: val[i+1:i+seq_length+1] for key, val in notes.items()}
        inputs.append(np.stack(list(input_seq.values()), axis=1))
        targets.append(np.stack(list(target_seq.values()), axis=1))

    return np.array(inputs), np.array(targets)

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dense(3)
    ])
    model.compile(loss="mse", optimizer="adam")
    return model
