# import os
# import pandas as pd
# import librosa
# import numpy as np
# import csv
# from pydub import AudioSegment
# from tqdm import tqdm
# import multiprocessing as mp

# # Constants
# metadata_file = 'dataset.csv'  # Path to your dataset file
# mfcc_output_folder = 'mfcc_files'
# os.makedirs(mfcc_output_folder, exist_ok=True)
# wav_output_folder = 'wav_files'
# os.makedirs(wav_output_folder, exist_ok=True)
# n_mfcc = 13  # Number of MFCC features

# def convert_mp3_to_wav(mp3_path, wav_path):
#     audio = None
#     try:
#         audio = AudioSegment.from_file(mp3_path, format='mp3')
#     except:
#         audio = AudioSegment.from_file(mp3_path, format='mp4')
#     if audio is None:
#         return False
#     os.makedirs(os.path.dirname(wav_path), exist_ok=True)  # Create directory if not exists
#     audio.export(wav_path, format='wav')
#     return True

# def extract_mfcc_features(wav_path, n_mfcc=13):
#     try:
#         audio, sample_rate = librosa.load(wav_path, sr=None)
#         mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
#         return mfcc_features
#     except Exception as e:
#         print(f"Error processing {wav_path}: {e}")
#         return None

# def save_mfcc_to_csv(mfcc_features, csv_file_path):
#     try:
#         os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)  # Create directory if not exists
#         num_mfcc = mfcc_features.shape[0]
#         num_frames = mfcc_features.shape[1]
#         with open(csv_file_path, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             header = [f'MFCC_{i+1}' for i in range(num_mfcc)]
#             writer.writerow(['Frame'] + header)
#             for i in range(num_frames):
#                 writer.writerow([i] + mfcc_features[:, i].tolist())
#     except Exception as e:
#         print(f"Error saving {csv_file_path}: {e}")

# def process_audio_file(file_info):
#     file_path, wav_file, csv_file = file_info
#     if os.path.exists(file_path):
#         if convert_mp3_to_wav(file_path, wav_file):
#             mfcc_features = extract_mfcc_features(wav_file)
#             if mfcc_features is not None:
#                 save_mfcc_to_csv(mfcc_features, csv_file)

# def create_dataset_and_process_files(metadata_file):
#     df = pd.read_csv(metadata_file)
    
#     # Prepare list for file processing
#     file_info_list = []
#     for index, row in df.iterrows():
#         mp3_path = row['Audio File']
#         relative_dir = os.path.dirname(mp3_path)  # Get folder structure of the mp3 file
#         wav_path = os.path.join(wav_output_folder, relative_dir, os.path.basename(mp3_path).replace('.mp3', '.wav'))
#         mfcc_csv_path = os.path.join(mfcc_output_folder, relative_dir, os.path.basename(mp3_path).replace('.mp3', '.csv'))
#         file_info_list.append((mp3_path, wav_path, mfcc_csv_path))
    
#     # Write the dataset mapping to a new CSV file
#     dataset_mapping_file = 'mfcc_included_dataset.csv'
#     with open(dataset_mapping_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Audio File', 'MFCC CSV File'])
#         for file_info in file_info_list:
#             if os.path.exists(file_info[0]):  # Check if the audio file exists
#                 writer.writerow([file_info[0], file_info[2]])
    
#     # Process files in parallel
#     with tqdm(total=len(file_info_list), desc="Processing Audio Files", unit="file") as pbar:
#         with mp.Pool(mp.cpu_count()) as pool:
#             for _ in tqdm(pool.imap_unordered(process_audio_file, file_info_list), total=len(file_info_list)):
#                 pbar.update()

# create_dataset_and_process_files(metadata_file)

# print(f"MFCC feature extraction complete. Files saved in the {mfcc_output_folder} folder.")
# print(f"WAV files saved in the {wav_output_folder} folder.")
# print(f"Dataset mapping saved in mfcc_included_dataset.csv.")

import os
import pandas as pd
import librosa
import numpy as np
import csv
from pydub import AudioSegment
from tqdm import tqdm
import multiprocessing as mp
import noisereduce as nr  # Import for spectral gating denoising

# Constants
metadata_file = 'dataset.csv'  # Path to your dataset file
mfcc_output_folder = 'mfcc_files'
os.makedirs(mfcc_output_folder, exist_ok=True)
wav_output_folder = 'wav_files'
os.makedirs(wav_output_folder, exist_ok=True)
n_mfcc = 13  # Number of MFCC features
threshold = 0.1  # Threshold for envelope function

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = None
    try:
        audio = AudioSegment.from_file(mp3_path, format='mp3')
    except:
        audio = AudioSegment.from_file(mp3_path, format='mp4')
    if audio is None:
        return False
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)  # Create directory if not exists
    audio.export(wav_path, format='wav')
    return True

def normalize_audio(y):
    """Normalize audio to the range of -1.0 to 1.0."""
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

def trim_audio(y, sr):
    """Trim the audio to remove leading and trailing silence."""
    trimmed_audio, _ = librosa.effects.trim(y)
    return trimmed_audio

def envelope(y, rate, threshold):
    mask = []
    y_abs = pd.Series(y).apply(np.abs)  # Convert to absolute values
    y_mean = y_abs.rolling(window=int(rate/10), min_periods=1, center=True).mean()  # Rolling mean
    for mean in y_mean:  # Apply thresholding
        mask.append(mean > threshold)  # Boolean mask
    return np.array(mask)

def spectral_gating(y, sr):
    """Apply spectral gating for noise reduction."""
    return nr.reduce_noise(y=y, sr=sr)

def extract_mfcc_features(wav_path, n_mfcc=13, threshold=0.02):
    try:
        audio, sample_rate = librosa.load(wav_path, sr=None)

        # Normalize the audio signal
        audio = normalize_audio(audio)

        # Trim the audio to remove leading and trailing silence
        audio = trim_audio(audio, sample_rate)

        # Apply spectral gating for noise reduction
        audio = spectral_gating(audio, sample_rate)

        # Apply the envelope function to filter the signal
        mask = envelope(audio, sample_rate, threshold)
        cleaned_audio = audio[mask]  # Filter the signal using the mask

        # Check if the cleaned audio is long enough for MFCC extraction
        if len(cleaned_audio) < 512:  # Minimum length for FFT
            print(f"Warning: Cleaned audio for {wav_path} is too short for MFCC extraction.")
            return None

        # Extract MFCC features from the cleaned audio
        mfcc_features = librosa.feature.mfcc(y=cleaned_audio, sr=sample_rate, n_mfcc=n_mfcc)
        return mfcc_features
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def save_mfcc_to_csv(mfcc_features, csv_file_path):
    try:
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)  # Create directory if not exists
        num_mfcc = mfcc_features.shape[0]
        num_frames = mfcc_features.shape[1]
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f'MFCC_{i+1}' for i in range(num_mfcc)]
            writer.writerow(['Frame'] + header)
            for i in range(num_frames):
                writer.writerow([i] + mfcc_features[:, i].tolist())
    except Exception as e:
        print(f"Error saving {csv_file_path}: {e}")

def process_audio_file(file_info):
    file_path, wav_file, csv_file = file_info
    if os.path.exists(file_path):
        if convert_mp3_to_wav(file_path, wav_file):
            mfcc_features = extract_mfcc_features(wav_file)
            if mfcc_features is not None:
                save_mfcc_to_csv(mfcc_features, csv_file)

def create_dataset_and_process_files(metadata_file):
    df = pd.read_csv(metadata_file)
    
    # Prepare list for file processing
    file_info_list = []
    for index, row in df.iterrows():
        mp3_path = row['Audio File']
        relative_dir = os.path.dirname(mp3_path)  # Get folder structure of the mp3 file
        wav_path = os.path.join(wav_output_folder, relative_dir, os.path.basename(mp3_path).replace('.mp3', '.wav'))
        mfcc_csv_path = os.path.join(mfcc_output_folder, relative_dir, os.path.basename(mp3_path).replace('.mp3', '.csv'))
        file_info_list.append((mp3_path, wav_path, mfcc_csv_path))
    
    # Process files in parallel
    with tqdm(total=len(file_info_list), desc="Processing Audio Files", unit="file") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(process_audio_file, file_info_list), total=len(file_info_list)):
                pbar.update()

create_dataset_and_process_files(metadata_file)

print(f"MFCC feature extraction complete. Files saved in the {mfcc_output_folder} folder.")
print(f"WAV files saved in the {wav_output_folder} folder.")
