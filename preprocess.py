import pretty_midi
import numpy as np

def midi_to_notes(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = sorted(instrument.notes, key=lambda note: note.start)

    pitches = [note.pitch for note in notes]
    starts = [note.start for note in notes]
    ends = [note.end for note in notes]
    durations = [end - start for start, end in zip(starts, ends)]
    steps = [start - starts[i - 1] if i > 0 else 0 for i, start in enumerate(starts)]

    return {
        "pitch": np.array(pitches),
        "step": np.array(steps),
        "duration": np.array(durations),
    }
