import numpy as np
from utils.audio_processing import process_audio, align_audio, reconstruct_audio
from utils.plotting import plot_time_domain_array, plot_frequency_domain_array, plot_frequency_domain
from config import FC1, FC2, FC3, LC1, HC1, LC2, HC2, LC3, HC3


def process():
    audio1, filtered1, mod1, mod_filtered1 = process_audio(FC1, 'input1')
    audio2, filtered2, mod2, mod_filtered2 = process_audio(FC2, 'input2')
    audio3, filtered3, mod3, mod_filtered3 = process_audio(FC3, 'input3')
    plot_time_domain_array([audio1, audio2, audio3], "Original", 1)
    plot_frequency_domain_array([audio1, audio2, audio3], "Original", 2)
    plot_frequency_domain_array([filtered1, filtered2, filtered3], 'LPF', 3)
    plot_frequency_domain_array([mod1, mod2, mod3], 'Modulation', 4)
    plot_frequency_domain_array(
        [mod_filtered1, mod_filtered2, mod_filtered3], 'Modulation_LPF', 5)
    return mod_filtered1, mod_filtered2, mod_filtered3


def combine(audio_list):
    aligned_audio = align_audio(audio_list)
    combined_audio = np.sum(aligned_audio, axis=0)
    plot_frequency_domain(combined_audio, 'Full_Spectrum', 6)
    return combined_audio


def reconstruct(combined):
    bf1, demod1, final1 = reconstruct_audio(combined, LC1, HC1, FC1, 'output1')
    bf2, demod2, final2 = reconstruct_audio(combined, LC2, HC2, FC2, 'output2')
    bf3, demod3, final3 = reconstruct_audio(combined, LC3, HC3, FC3, 'output3')
    plot_frequency_domain_array([bf1, bf2, bf3], 'BPF', 7)
    plot_frequency_domain_array([demod1, demod2, demod3], 'Demodulation', 8)
    plot_time_domain_array([final1, final2, final3], 'Final', 9)


def main():

    print("👋 Welcome to the Audio Signal Processing System!")
    print("-----------------------------------")

    print("🔄 Processing Audio Signals.....")
    audio_list = process()

    print("🔄 Combining Audio Signals.....")
    combined_audio = combine(audio_list)

    print("🔄 Reconstructing Audio Signals.....")
    reconstruct(combined_audio)

    print("-----------------------------------")
    print("✅ Done! Check the output folder for the audio files and images :)")


if __name__ == "__main__":
    main()
