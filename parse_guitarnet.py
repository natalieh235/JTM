import os
import jams
import librosa
import soundfile as sf
import numpy as np

import torch

def resample_audio(input_path, output_path, target_sr=16000):
    audio, sr = librosa.load(input_path, sr=None)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, audio_resampled, target_sr)

def normalize_audio(audio):
    return (audio - audio.mean()) / (audio.std() + 1e-8)

def segment_audio(audio, segment_length):
    return [audio[i:i+segment_length] for i in range(0, len(audio), segment_length)]

def preprocess_guitarset(input_dir, output_dir, target_sr=16000, segment_length=20480):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            # Load and resample
            file_path = os.path.join(input_dir, file_name)
            audio, sr = librosa.load(file_path, sr=None)
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Normalize
            audio_normalized = (audio_resampled - audio_resampled.mean()) / (audio_resampled.std() + 1e-8)

            # Segment audio
            segments = [
                audio_normalized[i:i + segment_length]
                for i in range(0, len(audio_normalized), segment_length)
                if len(audio_normalized[i:i + segment_length]) == segment_length
            ]

            # Save each segment
            base_name = os.path.splitext(file_name)[0]
            for i, segment in enumerate(segments):
                segment_path = os.path.join(output_dir, f"{base_name}_seg{i}.wav")
                sf.write(segment_path, segment, target_sr)


def process_guitarset_labels(jams_path, audio_length, window_size_ms, sampling_rate, n_notes=88):
    """
    Processes a GuitarSet JAMS file and aligns the note annotations with downsampled audio windows.
    
    Args:
    - jams_path (str): Path to the JAMS file.
    - audio_length (int): Length of the corresponding audio (in samples).
    - window_size_ms (float): Window size in milliseconds.
    - sampling_rate (int): Sampling rate of the audio (e.g., 16kHz).
    - n_notes (int): Number of notes (e.g., 88 for piano roll format).
    
    Returns:
    - torch.Tensor: Transcription tensor with shape (n_windows, n_notes + 1).
    """
    # Load JAMS file
    jam = jams.load(jams_path)
    note_data = jam.search(namespace='chord')  # Extract the first note_hz annotation
    notes = note_data.to_dataframe()

    # Convert start times and durations to seconds
    notes['start_time'] = notes['time']
    notes['end_time'] = notes['time'] + notes['duration']

    # Compute window size in seconds
    window_size_s = window_size_ms / 1000

    # Calculate number of windows
    downsampling = window_size_s * sampling_rate
    n_windows = int(np.ceil(audio_length / downsampling))

    # Initialize transcription tensor
    transcription = torch.zeros(n_windows, n_notes + 1)

    # Map note frequencies to MIDI note numbers
    notes['note'] = librosa.hz_to_midi(notes['value']).astype(int)

    # Align notes with windows
    for i in range(n_windows):
        start = i * window_size_s
        end = (i + 1) * window_size_s

        # Get notes active during this window
        active_notes = notes[(notes['start_time'] < end) & (notes['end_time'] > start)]['note'].values

        if len(active_notes) == 0:
            transcription[i, 0] = 1  # Mark as silence
        else:
            for note in active_notes:
                if 0 <= note < n_notes:
                    transcription[i, note] = 1

    return transcription

# Example usage
# input_directory = "path_to_guitarset_wavs"
# output_directory = "path_to_preprocessed_wavs"
# preprocess_guitarset(input_directory, output_directory)
chord_vocab = {"N": 0}  # Start with "No chord" as the first class
root_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
chord_types = [
    "maj", "min", "aug", "maj6", "min6", "7", "maj7", "min7", "dim7", "hdim7",
    "9", "maj9", "min9", "11", "sus2", "sus4", "maj/3", "maj/5", "min/b3",
    "min/5", "7/3", "7/5", "7/b7", "maj7/3", "maj7/5", "maj7/7", "min7/b3",
    "min7/5", "min7/b7", "dim7/b3", "dim7/b5", "dim7/bb7", "hdim7/b3", "hdim7/b5", "hdim7/b7"
]

# Populate the chord vocabulary
for root in root_notes:
    for chord_type in chord_types:
        chord_name = f"{root}:{chord_type}"
        chord_vocab[chord_name] = len(chord_vocab)

# Print the size of the vocabulary
print(f"Chord vocabulary size: {len(chord_vocab)}")