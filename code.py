
import numpy as np
import tensorflow as tf
import pandas as pd
import collections
import fluidsynth
import glob
import pretty_midi
from IPython import display
from typing import Dict, List, Optional, Sequence, Tuple
import zipfile
import os

# Destination folder
output_folder = "music-midi-dataset"
os.makedirs(output_folder, exist_ok=True)

# Extract ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

print("Extraction complete.")

sampling_rate = 44100

def display_audio(pm, seconds=30):
	waveform = pm.fluidsynth(fs=sampling_rate)
  # Take a sample of the generated waveform to mitigate kernel resets
	waveform_short = waveform[:seconds*sampling_rate]
	return display.Audio(waveform_short, rate=sampling_rate)

pm = pretty_midi.PrettyMIDI()
# Create an instrument instance and add it to the PrettyMIDI object
instrument = pretty_midi.Instrument(program=0, is_drum=False, name='acoustic grand piano') 
pm.instruments.append(instrument)
print(pm.instruments)
instrument = pm.instruments[0]
def midi_to_notes(midi_file):
	pm = pretty_midi.PrettyMIDI(midi_file)
	instrument = pm.instruments[0]
	notes = collections.defaultdict(list)
	sorted_notes = sorted(instrument.notes , key=lambda note:note.start)
	prev_start = sorted_notes[0].start

	for note in sorted_notes:
		start = note.start
		end = note.end
		notes["pitch"].append(note.pitch)
		notes["start"].append(start)
		notes["end"].append(end)
		notes["step"].append(start - prev_start)
		notes["duration"].append(end - start)
		prev_start = start
	return pd.DataFrame({name:np.array(value) for name,value in notes.items()})

raw_notes = midi_to_notes('C:/Users/Manjula/Downloads/archive/midi_dataset/midi_dataset/x (43).mid')
note_names = np.vectorize(pretty_midi.note_number_to_name)
sample_note_names = note_names(raw_notes["pitch"])
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import os

directory = r'C:\Users\Manjula\Downloads\archive\midi_dataset\midi_dataset'  # Adjust as necessary
filenames = glob.glob(os.path.join(directory, '*.mid'))  # Adjust path as needed


num_files = 5
all_notes = []

for f in filenames[:num_files]:
    notes = midi_to_notes(f)
    if not notes.empty:
        all_notes.append(notes)
    else:
        print(f"Skipping empty file: {f}")

if all_notes:
    all_notes_df = pd.concat(all_notes, ignore_index=True)
    print("Concatenated DataFrame:")
    print(all_notes_df)
else:
    print("No valid notes to concatenate.")

if 'all_notes_df' in locals():
    key_order = ["pitch", "step", "duration"]
    train_notes = np.stack([all_notes_df[key].values for key in key_order], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    print(notes_ds.element_spec)
else:
    print("Skipping the dataset creation as no valid DataFrame was available.")
seq_length = 20
vocab_size = 128

def create_sequences(train_notes, seq_length, vocab_size=128):
    sequences = []
    targets = []
    num_seq = train_notes.shape[0] - seq_length
    for i in range(num_seq):
        sequence = train_notes[i:i+seq_length - 1, :] / [vocab_size, 1, 1]
        target = train_notes[i + seq_length] / vocab_size
        sequences.append(sequence)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(sequences.shape, targets.shape)

    dataset = tf.data.Dataset.from_tensor_slices(
        (sequences, {
            "pitch": targets[:, 0],
            "step": targets[:, 1],
            "duration": targets[:, 2]
        })
    )
    return dataset

seq_ds = create_sequences(train_notes, seq_length=21, vocab_size=vocab_size)
batch_size = 64
buffer_size = 5000
train_ds = seq_ds.shuffle(buffer_size).batch(batch_size)
print(train_ds.element_spec)
layer = tf.keras.layers
learning_rate = 0.005

input_data = tf.keras.Input(shape=(seq_length, 3))
x = layer.LSTM(128)(input_data)

outputs = {
    "pitch": tf.keras.layers.Dense(64, name="pitch")(x),
    "step": tf.keras.layers.Dense(1, name="step")(x),
    "duration": tf.keras.layers.Dense(1, name="duration")(x),
}

model = tf.keras.Model(input_data, outputs)

loss = {
    "pitch": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "step": tf.keras.losses.MeanSquaredError(),
    "duration": tf.keras.losses.MeanSquaredError(),
}

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    loss=loss,
    loss_weights={
        "pitch": 0.05,
        "step": 1.0,
        "duration": 1.0,
    },
    optimizer=optimizer
)

model.summary()
model.fit(train_ds, epochs=10)

hist = model.predict(train_ds)
print(hist["duration"].shape)
def predict_next_note(notes, keras_model, temperature):
    assert temperature > 0
    inputs = np.expand_dims(notes, 0)
    predictions = keras_model.predict(inputs, verbose=0)  # Suppress output
    pitch_logits = predictions["pitch"]
    step = predictions["step"]
    duration = predictions["duration"]
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
    return int(pitch), float(step), float(duration)
# Start with the last sequence from training data
seed = train_notes[:seq_length]
generated_notes = []
temperature = 1.0  # Control randomness (try 0.7 to 1.2)

for i in range(100):  # Generate 100 notes
    pitch, step, duration = predict_next_note(seed, model, temperature)
    generated_notes.append({
        "pitch": pitch,
        "step": step,
        "duration": duration
    })

    # Append to seed and keep last `seq_length` items
    next_note = np.array([[pitch / vocab_size, step, duration]])
    seed = np.concatenate([seed[1:], next_note], axis=0)

# Convert to DataFrame
generated_notes_df = pd.DataFrame(generated_notes)
# Save and play MIDI
out_file = 'gfgmusicgenerate.mid'
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

out_pm = notes_to_midi(
    generated_notes_df, out_file=out_file, instrument_name=instrument_name
)

display_audio(out_pm, 30)  # Play 30 seconds