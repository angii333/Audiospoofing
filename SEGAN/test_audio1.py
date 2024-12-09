import os
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf  # For handling .flac files
from torch.autograd import Variable
from tqdm import tqdm
import glob
from data_preprocess import slice_signal, window_size, sample_rate
from model import Generator
from utils import emphasis

# List of input folders containing noisy .flac audio files
noisy_audio_folders = [
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_airport_0dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_airport_5dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_airport_10dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_airport_15dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_cafeteria_0dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_cafeteria_5dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_cafeteria_10dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_cafeteria_15dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_bable_0dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_bable_5dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_bable_10dB/",
    # "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_bable_15dB/",
#     "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_resto_0dB/",
#     "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_resto_5dB/",
#     "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_resto_10dB/",
#     "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_resto_15dB/",
#     "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_park_0dB/",
#     "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_park_5dB/",
#     "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_park_10dB/",
#     "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_park_15dB/"
]
# noisy_audio_folder = "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_cafeteria_0dB/"

# Path to the base folder for saving enhanced audio files
enhanced_audio_base_folder = "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_SEnoisyeval/"
# Ensure the output directory exists
os.makedirs(enhanced_audio_base_folder, exist_ok=True)



# Load SEGAN generator model
EPOCH_NAME = "generator-80.pkl"  # Replace with your generator epoch file
generator = Generator()
generator.load_state_dict(torch.load(f'epochs/{EPOCH_NAME}', map_location='cpu'))
if torch.cuda.is_available():
    generator.cuda()

# Loop through each input folder
for noisy_audio_folder in noisy_audio_folders:
    # Get the folder name (e.g., "recordings_audio_cafeteria_0dB")
    folder_name = os.path.basename(os.path.normpath(noisy_audio_folder))
    
    # Create the corresponding output folder for enhanced files
    enhanced_audio_folder = os.path.join(enhanced_audio_base_folder, folder_name)
    os.makedirs(enhanced_audio_folder, exist_ok=True)
    
    # Loop through each .flac file in the noisy audio folder
    for filename in os.listdir(noisy_audio_folder):
        if filename.endswith(".flac"):
            # Full path to the .flac file
            noisy_audio_path = os.path.join(noisy_audio_folder, filename)

            # Load the .flac audio file
            noisy_signal, sample_rate = sf.read(noisy_audio_path)

            # Slice the signal into smaller chunks
            noisy_slices = slice_signal(noisy_signal, window_size, 1, sample_rate)
            enhanced_speech = []

            # Process each slice using the SEGAN generator
            for noisy_slice in tqdm(noisy_slices, desc=f'Processing {filename}'):
                z = nn.init.normal_(torch.Tensor(1, 1024, 8))  # Random noise for generator
                noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
                if torch.cuda.is_available():
                    noisy_slice, z = noisy_slice.cuda(), z.cuda()
                noisy_slice, z = Variable(noisy_slice), Variable(z)

                # Generate enhanced audio slice
                generated_speech = generator(noisy_slice, z).data.cpu().numpy()
                generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
                generated_speech = generated_speech.reshape(-1)
                enhanced_speech.append(generated_speech)

            # Combine enhanced slices into a single waveform
            enhanced_speech = np.array(enhanced_speech).reshape(1, -1)

            # Full path to save the enhanced audio file in .flac format
            enhanced_audio_path = os.path.join(enhanced_audio_folder, filename)

            # Save the enhanced audio as .flac
            sf.write(enhanced_audio_path, enhanced_speech.T, sample_rate)
            print(f"Enhanced audio saved to {enhanced_audio_path}")

print("All audio files have been processed and saved as .flac.")
# # Specify the name of the file you want to process
# file_to_process = "LA_E_1165041.flac"  # Replace with the exact file name

# # Full path to the .flac file
# noisy_audio_path = os.path.join(noisy_audio_folder, file_to_process)

# # Load and process the specified file
# if os.path.exists(noisy_audio_path):
#     # Load the .flac audio file
#     noisy_signal, sample_rate = sf.read(noisy_audio_path)
    
#     # Print the length of the signal and the window size
#     print(f"Length of the signal: {len(noisy_signal)} samples")
#     print(f"Window size: {window_size} samples")

#     # Slice the signal into smaller chunks
#     noisy_slices = slice_signal(noisy_signal, window_size, 1, sample_rate)
#     enhanced_speech = []

#     # Process each slice using the SEGAN generator
#     for noisy_slice in tqdm(noisy_slices, desc=f'Processing {file_to_process}'):
#         z = nn.init.normal_(torch.Tensor(1, 1024, 8))  # Random noise for generator
#         noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
#         if torch.cuda.is_available():
#             noisy_slice, z = noisy_slice.cuda(), z.cuda()
#         noisy_slice, z = Variable(noisy_slice), Variable(z)

#         # Generate enhanced audio slice
#         generated_speech = generator(noisy_slice, z).data.cpu().numpy()
#         generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
#         generated_speech = generated_speech.reshape(-1)
#         enhanced_speech.append(generated_speech)

#     # Combine enhanced slices into a single waveform
#     enhanced_speech = np.array(enhanced_speech).reshape(1, -1)

#     # Full path to save the enhanced audio file in .flac format
#     enhanced_audio_path = os.path.join(enhanced_audio_base_folder, file_to_process)

#     # Save the enhanced audio as .flac
#     sf.write(enhanced_audio_path, enhanced_speech.T, sample_rate)
#     print(f"Enhanced audio saved to {enhanced_audio_path}")
# else:
#     print(f"File {file_to_process} does not exist in {noisy_audio_folder}")
