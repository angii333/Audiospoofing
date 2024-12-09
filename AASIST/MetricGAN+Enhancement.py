import torch
import torchaudio
import os
#from speechbrain.lobes.models.metricgan import MetricGAN
from speechbrain.inference.enhancement import SpectralMaskEnhancement




# List of input folders containing noisy .flac audio files
noisy_audio_folders = [
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_airport_0dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_airport_5dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_airport_10dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_airport_15dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_cafeteria_5dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_cafeteria_10dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_cafeteria_15dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_bable_0dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_bable_5dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_bable_10dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_bable_15dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_resto_0dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_resto_5dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_resto_10dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_resto_15dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_park_0dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_park_5dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_park_10dB/",
    "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_noisyeval/recordings_audio_park_15dB/"
    ]
# Path to the folder where you want to save the enhanced audio files
enhanced_audio_base_folder = "/home/angelaanacin/Notebooks/LA/ASVspoof2019_LA_enhanoisyeval"

# Ensure the output directory exists
os.makedirs(enhanced_audio_base_folder, exist_ok=True)

# Initialize the MetricGAN+ model
model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank", savedir="pretrained_models/metricgan-plus-voicebank",)

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
            waveform, sample_rate = torchaudio.load(noisy_audio_path)

            # Calculate the lengths tensor (use the full duration as 1.0)
            lengths = torch.tensor([1.0])

            # Apply the MetricGAN+ model to denoise the audio
            with torch.no_grad():  # Disable gradient calculation
                enhanced_waveform = model.enhance_batch(waveform, lengths)

            # Full path to save the enhanced audio file in .flac format
            enhanced_audio_path = os.path.join(enhanced_audio_folder, filename)

            # Save the enhanced audio as .flac
            torchaudio.save(enhanced_audio_path, enhanced_waveform, sample_rate, format="flac")
            print(f"Enhanced audio saved to {enhanced_audio_path}")

print("All audio files have been processed and saved as .flac.")
