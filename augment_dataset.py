import pandas as pd
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, speedup
import os
from tqdm import tqdm
import librosa

# Load original dataset
data = pd.read_csv('dataset.csv')

# Define augmentation parameters
output_audio_dir = './Augmented/Audio/'
output_mfcc_dir = './Augmented/Mfcc/'

# Ensure output directories exist
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(output_mfcc_dir, exist_ok=True)

# Augmentation function for audio with file skipping
def augment_audio(file_path, output_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        # Check if the augmented audio file already exists
        if os.path.exists(output_path):
            print(f"Audio file {output_path} already exists.")
            return
        
        # Try to load the file as an mp3 or mp4
        audio = None
        try:
            audio = AudioSegment.from_file(file_path, format='mp3')  # Try mp3 format first
        except:
            audio = AudioSegment.from_file(file_path, format='mp4')  # Try mp4 format if mp3 fails
        
        # Augment audio
        augmented_audio = normalize(audio)  # Normalize the audio
        augmented_audio = speedup(augmented_audio, playback_speed=1.1)  # Speed up the audio slightly
        augmented_audio.export(output_path, format='wav')
        print(f"Augmented audio file saved to {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to generate MFCC features from an audio file
def generate_mfcc(audio_file_path, mfcc_file_path):
    if not os.path.exists(audio_file_path):
        print(f"Audio file {audio_file_path} does not exist.")
        return

    # Check if MFCC file already exists
    if os.path.exists(mfcc_file_path):
        print(f"MFCC file {mfcc_file_path} already exists.")
        return

    # Load audio file
    y, sr = librosa.load(audio_file_path, sr=None)

    # Compute MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Save MFCC to file
    np.savetxt(mfcc_file_path, mfccs, delimiter=',')
    print(f"MFCC file saved to {mfcc_file_path}")

# Function to create synthetic samples with tqdm progress bar (only for word balancing)
def create_synthetic_samples(df, group_col, target_counts, output_audio_dir, output_mfcc_dir):
    augmented_data = []
    
    # Group by the specified column (Word)
    grouped = df.groupby(group_col)
    
    # Use tqdm to show progress over the words
    for category, group in tqdm(grouped, desc=f'Processing {group_col}', total=len(grouped)):
        num_samples = len(group)
        target_count = target_counts.get(category, 0)
        
        # If the word is underrepresented
        if num_samples < target_count:
            augmentation_factor = target_count - num_samples
            augmented_rows = []
            
            for _ in tqdm(range(augmentation_factor), desc=f'Augmenting {category}', leave=False):
                row = group.sample(n=1).iloc[0]
                new_audio_file = f"{output_audio_dir}aug_{np.random.randint(10000)}_{row['Audio File'].split('/')[-1]}"
                new_mfcc_file = f"{output_mfcc_dir}aug_{np.random.randint(10000)}_{row['Mfcc File'].split('/')[-1]}"
                
                # Augment the audio if it doesn't already exist
                augment_audio(row['Audio File'], new_audio_file)
                
                # Generate the MFCC for the augmented audio if it doesn't already exist
                generate_mfcc(new_audio_file, new_mfcc_file)
                
                # Create a new augmented row
                new_row = row.copy()
                new_row['Audio File'] = new_audio_file
                new_row['Mfcc File'] = new_mfcc_file
                augmented_rows.append(new_row)
            
            augmented_data.extend(augmented_rows)
    
    return pd.DataFrame(augmented_data)

# Define target counts for Word categories (balancing the word column)
word_target_count = data['Word'].value_counts().max()
target_counts_word = {word: word_target_count for word in data['Word'].unique()}

# Augment data based on Word category
augmented_data_word = create_synthetic_samples(data, 'Word', target_counts_word, output_audio_dir, output_mfcc_dir)

# Combine original and augmented data
final_df = pd.concat([data, augmented_data_word], ignore_index=True)

# Save the final dataset
final_df.to_csv('augmented_word_balanced_dataset.csv', index=False)
print("Augmentation complete and saved to 'augmented_word_balanced_dataset.csv'.")

