import os
import tempfile
import torchaudio
import librosa
import soundfile as sf
from pydub import AudioSegment
from audiomentations import Compose, Gain, Normalize, HighPassFilter, LowPassFilter

import replicate  # for input/output support

# Genre-specific mastering chains
GENRE_CHAINS = {
    "hiphop": Compose([
        Gain(min_gain_db=4, max_gain_db=5),
        Normalize(p=1.0),
        HighPassFilter(min_cutoff_freq=40.0, max_cutoff_freq=60.0, p=1.0),
        LowPassFilter(min_cutoff_freq=16000.0, max_cutoff_freq=18000.0, p=1.0),
    ]),
    "pop": Compose([
        Gain(min_gain_db=3, max_gain_db=4),
        Normalize(p=1.0),
        HighPassFilter(min_cutoff_freq=80.0, max_cutoff_freq=100.0, p=1.0),
        LowPassFilter(min_cutoff_freq=17000.0, max_cutoff_freq=18000.0, p=1.0),
    ]),
    "edm": Compose([
        Gain(min_gain_db=6, max_gain_db=8),
        Normalize(p=1.0),
        HighPassFilter(min_cutoff_freq=30.0, max_cutoff_freq=50.0, p=1.0),
    ]),
    "rock": Compose([
        Gain(min_gain_db=5, max_gain_db=6),
        Normalize(p=1.0),
        HighPassFilter(min_cutoff_freq=70.0, max_cutoff_freq=90.0, p=1.0),
    ]),
    "jazz": Compose([
        Gain(min_gain_db=2, max_gain_db=3),
        Normalize(p=1.0),
        HighPassFilter(min_cutoff_freq=50.0, max_cutoff_freq=70.0, p=1.0),
    ]),
}


def load_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    return waveform[0].numpy(), sr


def save_audio(data, sample_rate, path):
    sf.write(path, data, sample_rate)


def process_audio(audio_path, genre):
    samples, sr = load_audio(audio_path)
    augmenter = GENRE_CHAINS.get(genre, GENRE_CHAINS["hiphop"])
    processed_samples = augmenter(samples=samples, sample_rate=sr)

    temp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    save_audio(processed_samples, sr, temp_out.name)
    return temp_out.name


# Replicate's predict function
def predict(audio_input, genre="hiphop"):
    output_path = process_audio(audio_input, genre)
    return output_path
