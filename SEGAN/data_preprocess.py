import os

import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf

clean_train_folder = 'data/clean_trainset_56spk_wav'
noisy_train_folder = 'data/noisy_trainset_56spk_wav'
clean_test_folder = 'data/clean_testset_wav'
noisy_test_folder = 'data/noisy_testset_wav'
serialized_train_folder = 'data/serialized_train_data'
serialized_test_folder = 'data/serialized_test_data'
window_size = 16000  # about 1 second of samples
sample_rate = 16000



# def slice_signal(audio_input, window_size, stride, sample_rate):
#     """
#     Helper function for slicing the audio signal
#     by window size and sample rate with [1-stride] percent overlap (default 50%).
#     If the input is a file path, it reads the file. Otherwise, it assumes the input is an audio array.
#     """
#     if isinstance(audio_input, str):  # If input is a file path
#         wav, sr = sf.read(audio_input)
#         if sr != sample_rate:
#             raise ValueError(f"Sample rate mismatch: {sr} != {sample_rate}")
#     elif isinstance(audio_input, np.ndarray):  # If input is already an audio array
#         wav = audio_input
#     else:
#         raise TypeError(f"Invalid input type: {type(audio_input)}. Expected file path or audio array.")
    
#     hop = int(window_size * stride)
#     slices = []
#     for end_idx in range(window_size, len(wav), hop):
#         start_idx = end_idx - window_size
#         slice_sig = wav[start_idx:end_idx]
#         slices.append(slice_sig)
#     return slices
def slice_signal(audio_input, window_size, stride, sample_rate):
    """
    Slices the audio signal into smaller chunks based on the window size and stride.
    Handles cases where the input signal length is smaller than the window size.
    """
    if isinstance(audio_input, str):  # If input is a file path
        wav, sr = sf.read(audio_input)
        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {sr} != {sample_rate}")
    elif isinstance(audio_input, np.ndarray):  # If input is already an audio array
        wav = audio_input
    else:
        raise TypeError(f"Invalid input type: {type(audio_input)}. Expected file path or audio array.")
    
    hop = int(window_size * stride)
    slices = []

    
    # Case 1: Signal length < window_size (pad with zeros)
    if len(wav) < window_size:
        print(f"Signal length ({len(wav)}) is smaller than window size ({window_size}). Padding...")
        padded_wav = np.pad(wav, (0, window_size - len(wav)), mode='constant')
        slices.append(padded_wav)

    # Case 2: Signal length == window_size
    elif len(wav) == window_size:
        print(f"Signal length ({len(wav)}) matches the window size ({window_size}). Using the signal directly.")
        slices.append(wav)

    # Case 3: Signal length > window_size (normal slicing)
    else:
        print(f"Signal length ({len(wav)}) is greater than window size ({window_size}). Slicing...")
        for end_idx in range(window_size, len(wav) + 1, hop):  # Ensure last slice is included
            start_idx = end_idx - window_size
            slice_sig = wav[start_idx:end_idx]
            slices.append(slice_sig)

        # Handle any remaining samples at the end of the signal
        if len(wav) % hop != 0:
            print("Remaining samples detected. Padding the last slice...")
            last_slice = wav[-window_size:]  # Take the last `window_size` samples
            slices.append(last_slice)

    return slices



def process_and_serialize(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    stride = 0.5  # 50% overlap

    if data_type == 'train':
        clean_folder = clean_train_folder
        noisy_folder = noisy_train_folder
        serialized_folder = serialized_train_folder
    else:
        clean_folder = clean_test_folder
        noisy_folder = noisy_test_folder
        serialized_folder = serialized_test_folder

    if not os.path.exists(serialized_folder):
        os.makedirs(serialized_folder)

    # Walk through the clean folder and process files
    for root, dirs, files in os.walk(clean_folder):
        if len(files) == 0:
            continue
        for filename in tqdm(files, desc=f'Serialize and down-sample {data_type} audios'):
            # Ensure we're processing only .flac files
            if not filename.endswith(".flac"):
                continue

            clean_file = os.path.join(clean_folder, filename)
            noisy_file = os.path.join(noisy_folder, filename)

            # Slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_file, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_file, window_size, stride, sample_rate)

            # Serialize - file format goes [original_file]_[slice_number].npy
            # Example: file1.flac_5.npy denotes 5th slice of file1.flac file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(serialized_folder, f"{filename}_{idx}.npy"), arr=pair)


def data_verify(data_type):
    """
    Verifies the length of each data after preprocessing.
    """
    if data_type == 'train':
        serialized_folder = serialized_train_folder
    else:
        serialized_folder = serialized_test_folder

    for root, dirs, files in os.walk(serialized_folder):
        for filename in tqdm(files, desc=f'Verify serialized {data_type} audios'):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print(f'Snippet length not {window_size}: {data_pair.shape[1]} instead')
                break


if __name__ == '__main__':
    process_and_serialize('train')
    data_verify('train')
    process_and_serialize('test')
    data_verify('test')