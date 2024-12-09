import os
import glob
import numpy as np
import soundfile as sf
import librosa

def add_noise(audio_folder, noise_folder, noise_sample, snr):
    """
    Add noise to an audio signal and save the noisy files.

    Parameters:
    - audio_folder (str): Directory with clean audio signals (FLAC format).
    - noise_folder (str): Directory where corrupted signals will be saved.
    - noise_sample (str): Path to a single noise sample file (FLAC format).
    - snr (float): Signal-to-Noise Ratio level to be used.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(noise_folder):
        os.makedirs(noise_folder)

    # List all .flac files in the audio folder
    audio_files = glob.glob(os.path.join(audio_folder, '*.flac'))

    # Load the noise sample
    noise, sr_noise = sf.read(noise_sample)

    for audio_file in audio_files:
        # Load the clean audio file
        clean, sr_clean = sf.read(audio_file)

        # Ensure that the noise and clean signal have the same sample rate
        if sr_clean != sr_noise:
             # Resample noise to match the sample rate of the clean signal
            noise = librosa.resample(y=noise, orig_sr=sr_noise, target_sr=sr_clean)
            sr_noise = sr_clean
            print(f"Resampled noise to match clean audio sample rate.")

        # Add noise to the clean signal
        noisy_signal = add_noise_to_signal(clean, noise, snr)

        # Normalize the noisy signal
        noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

        # Define the output file path
        output_file = os.path.join(noise_folder, os.path.basename(audio_file))
        
        # Save the noisy audio file
        sf.write(output_file, noisy_signal, sr_clean)

        print(f"Processed file saved to: {output_file}")

def add_noise_to_signal(clean_signal, noise, snr):
    """
    Add noise to the clean signal based on the desired SNR.

    Parameters:
    - clean_signal (numpy array): The clean audio signal.
    - noise (numpy array): The noise signal to be added.
    - snr (float): Desired Signal-to-Noise Ratio.

    Returns:
    - noisy_signal (numpy array): The clean signal with added noise.
    """
    # Calculate the power of the clean signal and noise
    clean_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise**2)

    # Calculate the required noise power for the desired SNR
    snr_linear = 10**(snr / 10.0)
    required_noise_power = clean_power / snr_linear

    # Scale the noise to match the required noise power
    noise = noise * np.sqrt(required_noise_power / noise_power)

    # Ensure that noise is the same length as the clean signal
    if len(noise) > len(clean_signal):
        noise = noise[:len(clean_signal)]
    elif len(noise) < len(clean_signal):
        noise = np.pad(noise, (0, len(clean_signal) - len(noise)))

    # Add noise to the clean signal
    noisy_signal = clean_signal + noise

    return noisy_signal

# Example usage:
# add_noise('/path/to/audio_folder', '/path/to/noise_folder', '/path/to/noise_sample.flac', 20)
