import os
import pickle
import random
import numpy as np
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import load_model


MODEL_PATH = "model"
GENERATED_PATH = "generated"
SEQUENCE_LENGTH = 100


def load_data():
    with open(os.path.join(MODEL_PATH, "notes.pkl"), "rb") as file:
        notes = pickle.load(file)

    with open(os.path.join(MODEL_PATH, "note_to_int.pkl"), "rb") as file:
        note_to_int = pickle.load(file)

    with open(os.path.join(MODEL_PATH, "int_to_note.pkl"), "rb") as file:
        int_to_note = pickle.load(file)

    return notes, note_to_int, int_to_note


def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-9) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)

    return np.random.choice(len(predictions), p=predictions)


def generate_notes(output_length=300, temperature=0.8):
    model_file = os.path.join(MODEL_PATH, "best_model.h5")

    if not os.path.exists(model_file):
        raise FileNotFoundError("Trained model not found. Run train_model.py first.")

    notes, note_to_int, int_to_note = load_data()
    model = load_model(model_file)

    n_vocab = len(set(notes))

    network_input = []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        sequence = notes[i:i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[n] for n in sequence])

    start_index = random.randint(0, len(network_input) - 1)
    pattern = network_input[start_index]

    prediction_output = []

    for _ in range(output_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)[0]

        index = sample_with_temperature(prediction, temperature)
        result = int_to_note[index]

        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output


def create_midi(prediction_output, output_file):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if "." in pattern or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            chord_notes = []

            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)

            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_file)


def generate_midi_file(output_length=300, temperature=0.8):
    os.makedirs(GENERATED_PATH, exist_ok=True)

    prediction_output = generate_notes(output_length, temperature)

    output_file = os.path.join(GENERATED_PATH, "generated_music.mid")
    create_midi(prediction_output, output_file)

    print(f"Generated MIDI saved: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_midi_file()