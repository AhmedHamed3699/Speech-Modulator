import numpy as np
from scipy.io import wavfile
from scipy import signal
from utils.filters import low_pass_filter, band_pass_filter
from config import TARGET_FS


def read_audio(filename):
    fs, audio = wavfile.read(filename)
    audio_resampled = signal.resample(audio, int(len(audio) * TARGET_FS / fs))
    return audio_resampled


def align_audio(audio_list):
    min_length = min(len(audio) for audio in audio_list)
    aligned_audio = [audio[:min_length] for audio in audio_list]
    return aligned_audio


def modulate_audio(audio, Fc):
    t = np.arange(len(audio)) / TARGET_FS
    return audio * np.cos(2 * np.pi * Fc * t)


def process_audio(Fc, input_name):
    audio = read_audio(f'../input/{input_name}.wav')
    filtered = low_pass_filter(audio)
    mod = modulate_audio(filtered, Fc)
    mod_filtered = low_pass_filter(mod, Fc, 200)  # 200 for sideband filtering
    return audio, filtered, mod, mod_filtered


def reconstruct_audio(combined_audio, low_cutoff, high_cutoff, Fc, output_name):
    bf = band_pass_filter(combined_audio, low_cutoff, high_cutoff)
    demod = modulate_audio(bf, Fc)
    final = low_pass_filter(demod)
    output_file = f'../output/voices/{output_name}.wav'
    wavfile.write(output_file, TARGET_FS, final.astype(np.int16))
    return bf, demod, final
