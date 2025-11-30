import numpy as np
import pretty_midi

def generate_notes(model, start_sequence, num_steps=50):
    generated = start_sequence.copy()
    input_seq = start_sequence[np.newaxis, ...]

    for _ in range(num_steps):
        prediction = model.predict(input_seq, verbose=0)
        next_note = prediction[0, -1]
        input_seq = np.append(input_seq[:, 1:, :], [[next_note]], axis=1)
        generated = np.append(generated, [next_note], axis=0)

    return generated

def notes_to_midi(generated_notes, output_file="output.mid"):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    start = 0
    for pitch, step, duration in generated_notes:
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=start + float(step),
            end=start + float(step) + float(duration)
        )
        instrument.notes.append(note)
        start = note.start

    pm.instruments.append(instrument)
    pm.write(output_file)
