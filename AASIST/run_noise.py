import os
import numpy as np
from scipy.io import wavfile
from add_noise import add_noise  # Make sure this function is defined in add_Noise.py

def process_audio(audiofolder, noisysample, noisefolder, snr):
    """
    Process the audio by adding noise to it.
    """
    if not os.path.exists(noisefolder):
        os.makedirs(noisefolder)
    add_noise(audiofolder, noisefolder, noisysample, snr)

def main():
    audiofolder = '/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_eval/flac/'
    
    snrs = [10, 15]
    noise_types = ['BABLE', 'CAFETERIA', 'AIRPORT', 'PARK', 'RESTO']
    reverb = ''  # No reverberation
    
    for snr in snrs:
        for noise_type in noise_types:
            noisysample = f'/home/angelaanacin/Notebooks/noise_add/noise/{noise_type}/ch01{reverb}.flac'
            noisefolder = f'./recordings_audio_{noise_type.lower()}_{snr}dB'
            process_audio(audiofolder, noisysample, noisefolder, snr)
    
    # If you need to add reverberation, you can use similar blocks as shown below
    
    # reverb_times = ['0.25', '0.48', '0.8']
    # for reverb in reverb_times:
    #     for snr in snrs:
    #         for noise_type in noise_types:
    #             noisysample = f'./noise/{noise_type}/ch01.flac'
    #             noisefolder = f'./recordings_audio_rev_{reverb}_{noise_type.lower()}_{snr}dB'
    #             process_audio(audiofolder, noisysample, noisefolder, snr)

if __name__ == "__main__":
    main()
