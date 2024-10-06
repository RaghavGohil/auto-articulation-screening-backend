import csv
import librosa
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = None
    try:
        audio = AudioSegment.from_file(mp3_path, format='mp3')
    except:
        audio = AudioSegment.from_file(mp3_path, format='mp4')
    if audio is None:
        return False
    audio.export(wav_path, format='wav')
    return True

def extract_mfcc_features(wav_path, n_mfcc=13):
    try:
        audio, sample_rate = librosa.load(wav_path, sr=None)
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return mfcc_features
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def save_mfcc_to_csv(mfcc_features, csv_file_path):
    try:
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

convert_mp3_to_wav('cat_test_3.mp3','cat_test_3.wav')
mfcc = extract_mfcc_features('cat_test_3.wav')
save_mfcc_to_csv(mfcc,'cat_mfcc_3')
