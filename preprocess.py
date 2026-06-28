import os
import glob
import pickle
from music21 import converter, instrument, note, chord


DATASET_PATH = "data/midi_files"
MODEL_PATH = "model"


def extract_notes():
    notes = []

    midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid")) + glob.glob(os.path.join(DATASET_PATH, "*.midi"))

    if not midi_files:
        raise FileNotFoundError("No MIDI files found. Add .mid or .midi files inside data/midi_files folder.")

    print(f"Found {len(midi_files)} MIDI files.")

    for file_path in midi_files:
        try:
            print(f"Processing: {file_path}")
            midi = converter.parse(file_path)

            parts = instrument.partitionByInstrument(midi)

            if parts:
                elements = parts.parts[0].recurse()
            else:
                elements = midi.flatten().notes

            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

        except Exception as e:
            print(f"Skipped file due to error: {file_path}")
            print(e)

    if len(notes) < 150:
        raise ValueError("Not enough notes extracted. Add more MIDI files to train the model properly.")

    os.makedirs(MODEL_PATH, exist_ok=True)

    with open(os.path.join(MODEL_PATH, "notes.pkl"), "wb") as file:
        pickle.dump(notes, file)

    print(f"Preprocessing complete. Total notes/chords extracted: {len(notes)}")
    print("Saved: model/notes.pkl")


if __name__ == "__main__":
    extract_notes()