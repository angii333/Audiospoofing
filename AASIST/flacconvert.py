from pydub import AudioSegment
import os

# Directory where your .wav files are located
input_directory = "/home/angelaanacin/Notebooks/noise_add/noise/RESTO"
# Directory where you want to save the .flac files
output_directory = "/home/angelaanacin/Notebooks/noise_add/noise/RESTO"


# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each .wav file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        # Full path to the input .wav file
        input_file = os.path.join(input_directory, filename)
        # Full path for the output .flac file
        output_file = os.path.join(output_directory, filename.replace(".wav", ".flac"))
        
        # Load the .wav file
        audio = AudioSegment.from_wav(input_file)
        
        # Export as .flac to the specified directory
        audio.export(output_file, format="flac")
