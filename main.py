import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile


# Function to read and resample audio
def read_and_resample_audio(filename, target_fs=54000):
    fs, audio = wavfile.read(filename)
    audio_resampled = signal.resample(audio, int(len(audio) * target_fs / fs))
    return audio_resampled, target_fs

def plot_time_domain(audio, fs, figure_name):
    plt.figure()
    plt.plot(audio)
    plt.title(f'{figure_name} - Time Domain')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.savefig(f'{figure_name}_time_domain.png')  # Save as PNG
    plt.close()  # Close the figure to save memory

def plot_frequency_domain(audio, fs, figure_name):
    plt.figure()
    # Perform FFT and plot the frequency spectrum
    spectrum = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/fs)
    plt.plot(freqs[:len(freqs)//2], np.abs(spectrum)[:len(spectrum)//2])
    plt.title(f'{figure_name} - Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig(f'{figure_name}_frequency_domain.png')  # Save as PNG
    plt.close()  # Close the figure to save memory

def plot_combined_spectrum(combined_audio, fs):
    plt.figure()
    spectrum = np.fft.fft(combined_audio)
    freqs = np.fft.fftfreq(len(combined_audio), 1/fs)
    plt.plot(freqs[:len(freqs)//2], np.abs(spectrum)[:len(spectrum)//2])
    plt.title('Combined Audio Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig('combined_audio_spectrum.png')  # Save as PNG
    plt.close()  # Close the figure to save memory


# Function to apply a low-pass filter
def apply_low_pass_filter(audio, fs, cutoff=3000, filter_order=50):
    lpf = signal.firwin(filter_order, cutoff, fs=fs)
    filtered_audio = signal.lfilter(lpf, 1.0, audio)
    return filtered_audio


# Function for modulation
def modulate_audio(audio, fs, Fc):
    t = np.arange(len(audio)) / fs
    return audio * np.cos(2 * np.pi * Fc * t)


# Function to apply a sideband filter after modulation
def apply_sideband_filter(audio, fs, Fc, filter_order=500):
    lpf = signal.firwin(filter_order, Fc, fs=fs)
    filtered_audio = signal.lfilter(lpf, 1.0, audio)
    return filtered_audio


# Function to apply a band-pass filter for demodulation
def apply_band_pass_filter(audio, fs, low_cutoff, high_cutoff, filter_order=50):
    bpf = signal.firwin(filter_order, [low_cutoff, high_cutoff], pass_zero=False, fs=fs)
    return signal.lfilter(bpf, 1.0, audio)

def align_audio_signals(audio_list):
    # Find the minimum length among the audio signals
    min_length = min(len(audio) for audio in audio_list)

    # Trim or pad each signal to the minimum length
    aligned_audio = [audio[:min_length] for audio in audio_list]
    return aligned_audio

def main():
    # Process input1
    audio1, fs = read_and_resample_audio('input1.wav')
    plot_time_domain(audio1, fs, "Audio1 Before Filter")
    plot_frequency_domain(audio1, fs, 'Audio1 Before Filter')

    filtered_audio1 = apply_low_pass_filter(audio1, fs)
    plot_frequency_domain(filtered_audio1, fs, 'Audio1 After Filter')

    Fc1 = 5000
    modulated_audio1 = modulate_audio(filtered_audio1, fs, Fc1)
    plot_frequency_domain(modulated_audio1, fs, 'Audio1 Amplitude Spectrum')

    filtered_audio1_after_mod = apply_sideband_filter(modulated_audio1, fs, Fc1)
    plot_frequency_domain(filtered_audio1_after_mod, fs, 'Audio1 After Sideband Filter')

    # Process input2
    audio2, fs = read_and_resample_audio('input2.wav')
    plot_time_domain(audio2, fs, "Audio2 Before Filter")
    plot_frequency_domain(audio2, fs, 'Audio2 Before Filter')

    filtered_audio2 = apply_low_pass_filter(audio2, fs)
    plot_frequency_domain(filtered_audio2, fs, 'Audio2 After Filter')

    Fc2 = 14000
    modulated_audio2 = modulate_audio(filtered_audio2, fs, Fc2)
    plot_frequency_domain(modulated_audio2, fs, 'Audio2 Amplitude Spectrum')

    filtered_audio2_after_mod = apply_sideband_filter(modulated_audio2, fs, Fc2)
    plot_frequency_domain(filtered_audio2_after_mod, fs, 'Audio2 After Sideband Filter')

    # Process input3
    audio3, fs = read_and_resample_audio('input3.wav')
    plot_time_domain(audio3, fs, "Audio3 Before Filter")
    plot_frequency_domain(audio3, fs, 'Audio3 Before Filter')

    filtered_audio3 = apply_low_pass_filter(audio3, fs)
    plot_frequency_domain(filtered_audio3, fs, 'Audio3 After Filter')

    Fc3 = 23000
    modulated_audio3 = modulate_audio(filtered_audio3, fs, Fc3)
    plot_frequency_domain(modulated_audio3, fs, 'Audio3 Amplitude Spectrum')

    filtered_audio3_after_mod = apply_sideband_filter(modulated_audio3, fs, Fc3)
    plot_frequency_domain(filtered_audio3_after_mod, fs, 'Audio3 After Sideband Filter')

    # Align the audio signals to the same length before combining them
    aligned_audio = align_audio_signals([filtered_audio1_after_mod, filtered_audio2_after_mod, filtered_audio3_after_mod])

    # Combine all processed audio signals
    combined_audio = np.sum(aligned_audio, axis=0)
    plot_combined_spectrum(combined_audio, fs)

    # Apply band-pass filter and demodulate for input1
    band_filtered_audio1 = apply_band_pass_filter(combined_audio, fs, 1000, 9000)
    plot_frequency_domain(band_filtered_audio1, fs, 'Audio1 Before Demodulation')

    demodulated_audio1 = modulate_audio(band_filtered_audio1, fs, Fc1)
    filtered_audio1_final = apply_low_pass_filter(demodulated_audio1, fs, cutoff=4000)
    plot_frequency_domain(filtered_audio1_final, fs, 'Audio1 Final After Demodulation')

    wavfile.write('output1.wav', fs, filtered_audio1_final.astype(np.int16))

    # Apply band-pass filter and demodulate for input2
    band_filtered_audio2 = apply_band_pass_filter(combined_audio, fs, 10000, 18000)
    plot_frequency_domain(band_filtered_audio2, fs, 'Audio2 Before Demodulation')

    demodulated_audio2 = modulate_audio(band_filtered_audio2, fs, Fc2)
    filtered_audio2_final = apply_low_pass_filter(demodulated_audio2, fs, cutoff=4000)
    plot_frequency_domain(filtered_audio2_final, fs, 'Audio2 Final After Demodulation')

    wavfile.write('output2.wav', fs, filtered_audio2_final.astype(np.int16))

    # Apply band-pass filter and demodulate for input3
    band_filtered_audio3 = apply_band_pass_filter(combined_audio, fs, 19000, 26999)
    plot_frequency_domain(band_filtered_audio3, fs, 'Audio3 Before Demodulation')

    demodulated_audio3 = modulate_audio(band_filtered_audio3, fs, Fc3)
    filtered_audio3_final = apply_low_pass_filter(demodulated_audio3, fs, cutoff=4000)
    plot_frequency_domain(filtered_audio3_final, fs, 'Audio3 Final After Demodulation')

    wavfile.write('output3.wav', fs, filtered_audio3_final.astype(np.int16))

# Call main function to execute the program
if __name__ == "__main__":
    main()
