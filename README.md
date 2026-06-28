# AI Music Generation Using LSTM

## Overview
This project generates new MIDI music using an LSTM deep learning model trained on MIDI note sequences.

## Features
- MIDI file preprocessing
- Note and chord extraction using music21
- LSTM-based sequence learning
- New MIDI music generation
- Flask web interface
- Download generated MIDI output

## Technologies Used
- Python
- TensorFlow / Keras
- music21
- Flask
- NumPy
- HTML / CSS

## Project Workflow
1. Collect MIDI files
2. Extract notes and chords
3. Convert music data into numerical sequences
4. Train LSTM model
5. Generate new note sequences
6. Convert generated output to MIDI

## How to Run
```bash
pip install -r requirements.txt
python preprocess.py
python train_model.py
python generate_music.py
python app.py
