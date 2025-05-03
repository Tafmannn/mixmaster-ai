import os
import subprocess
import tempfile
import torchaudio
import demucs.separate
from pydub import AudioSegment
import pyloudnorm as pyln

GENRE_PRESETS = {
    "hiphop": {"low_cut": 40, "high_cut": 16000, "compression_ratio": 4, "target_lufs": -8},
    "edm": {"low_cut": 30, "high_cut": 18000, "compression_ratio": 3, "target_lufs": -6},
    "pop": {"low_cut": 50, "high_cut": 17000, "compression_ratio": 2.5, "target_lufs": -10},
    "rock": {"low_cut": 60, "high_cut": 15000, "compression_ratio": 5, "target_lufs": -9}
}

def apply_dsp(stem_path, preset):
    audio = AudioSegment.from_file(stem_path)
    audio = audio.high_pass_filter(preset['low_cut']).low_pass_filter(preset['high_cut'])
    temp_out = tempfile.mktemp(suffix=".wav")
    audio.export(temp_out, format="wav")
    return temp_out

def predict(audio_file, genre="hiphop", target_lufs=-8.0):
    preset = GENRE_PRESETS[genre]
    work_dir = tempfile.mkdtemp()
    audio_path = os.path.join(work_dir, "input.wav")
    AudioSegment.from_file(audio_file).export(audio_path, format="wav")

    # Stem separation with Demucs
    demucs_output_dir = os.path.join(work_dir, "separated")
    subprocess.run(["python3", "-m", "demucs.separate", "-n", "htdemucs", "-o", demucs_output_dir, audio_path])

    stems_dir = os.path.join(demucs_output_dir, "htdemucs", "input")
    stem_files = []
    for file in os.listdir(stems_dir):
        stem_path = os.path.join(stems_dir, file)
        processed_path = apply_dsp(stem_path, preset)
        stem_files.append(processed_path)

    # Mix stems into one track
    final_mix = AudioSegment.silent(duration=0)
    for f in stem_files:
        final_mix = final_mix.overlay(AudioSegment.from_file(f))

    # Loudness normalization
    samples = final_mix.get_array_of_samples()
    meter = pyln.Meter(final_mix.frame_rate)
    loudness = meter.integrated_loudness(samples)
    final_mix = final_mix + (target_lufs - loudness)

    master_output_path = os.path.join(work_dir, "final_master.wav")
    final_mix.export(master_output_path, format="wav")

    return {"master_output": master_output_path, "stem_outputs": stem_files}