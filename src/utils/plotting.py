import matplotlib.pyplot as plt
import numpy as np
from config import TARGET_FS


def plot_time_domain_array(audio_list, figure_name, img_index):
    output = f'../output/images/{img_index}-{figure_name}_time_domain.png'
    fig, axs = plt.subplots(len(audio_list), 1, figsize=(15, 10))
    for idx, audio in enumerate(audio_list):
        axs[idx].plot(audio)
        axs[idx].set_title(f'{figure_name} {idx + 1} - Time Domain')
        axs[idx].set_xlabel('Samples')
        axs[idx].set_ylabel('Amplitude')
        axs[idx].grid(True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_frequency_domain_array(audio_list, figure_name, img_index):
    output = f'../output/images/{img_index}-{figure_name}_frequency_domain.png'
    fig, axs = plt.subplots(len(audio_list), 1, figsize=(15, 10))
    for idx, audio in enumerate(audio_list):
        spectrum = np.fft.fftshift(np.fft.fft(audio))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(audio), 1 / TARGET_FS))
        axs[idx].plot(freqs, np.abs(spectrum))
        axs[idx].set_title(f'{figure_name} {idx + 1} - Frequency Domain')
        axs[idx].set_xlabel('Frequency (Hz)')
        axs[idx].set_ylabel('Amplitude')
        axs[idx].grid(True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_frequency_domain(audio, figure_name, img_index):
    spectrum = np.fft.fftshift(np.fft.fft(audio))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(audio), 1 / TARGET_FS))
    output = f'../output/images/{img_index}-{figure_name}_frequency_domain.png'
    plt.figure(figsize=(15, 10))
    plt.plot(freqs, np.abs(spectrum))
    plt.title(f'{figure_name} - Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(output)
    plt.close()
