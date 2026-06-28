import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical


MODEL_PATH = "model"
SEQUENCE_LENGTH = 100


def load_notes():
    notes_file = os.path.join(MODEL_PATH, "notes.pkl")

    if not os.path.exists(notes_file):
        raise FileNotFoundError("notes.pkl not found. Run preprocess.py first.")

    with open(notes_file, "rb") as file:
        notes = pickle.load(file)

    return notes


def prepare_sequences(notes):
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    int_to_note = {number: note for number, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]

        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    if n_patterns == 0:
        raise ValueError("Not enough note sequences. Add more MIDI files.")

    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    network_input = network_input / float(n_vocab)

    network_output = to_categorical(network_output, num_classes=n_vocab)

    with open(os.path.join(MODEL_PATH, "note_to_int.pkl"), "wb") as file:
        pickle.dump(note_to_int, file)

    with open(os.path.join(MODEL_PATH, "int_to_note.pkl"), "wb") as file:
        pickle.dump(int_to_note, file)

    return network_input, network_output, n_vocab


def create_model(network_input, n_vocab):
    model = Sequential()

    model.add(
        LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        )
    )
    model.add(Dropout(0.3))

    model.add(LSTM(256))
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(n_vocab, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def train():
    notes = load_notes()
    network_input, network_output, n_vocab = prepare_sequences(notes)

    model = create_model(network_input, n_vocab)

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_PATH, "best_model.h5"),
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    )

    early_stop = EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True
    )

    print("Training started...")

    model.fit(
        network_input,
        network_output,
        epochs=50,
        batch_size=64,
        callbacks=[checkpoint, early_stop]
    )

    model.save(os.path.join(MODEL_PATH, "final_model.h5"))

    print("Training complete.")
    print("Saved model/best_model.h5 and model/final_model.h5")


if __name__ == "__main__":
    train()