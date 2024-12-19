from scipy import signal
from config import TARGET_FS


def low_pass_filter(audio, cutoff=4000, filter_order=50):
    lpf = signal.firwin(filter_order, cutoff, fs=TARGET_FS)
    filtered_audio = signal.lfilter(lpf, 1.0, audio)
    return filtered_audio


def band_pass_filter(audio, low_cutoff, high_cutoff, filter_order=50):
    bpf = signal.firwin(
        filter_order, [low_cutoff, high_cutoff], pass_zero=False, fs=TARGET_FS)
    return signal.lfilter(bpf, 1.0, audio)
