import numpy as np
import pandas as pd
import os
import librosa
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('word_prediction_model.keras')

# Function to load and preprocess audio file
def preprocess_audio(file_path, max_length):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)  # Use the original sample rate
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Adjust n_mfcc as needed
    # Pad or truncate MFCC to the maximum length
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    return mfcc

# Function to predict word from audio file
def predict_word(file_path):
    # Preprocess audio
    max_length = 100  # Define the maximum length based on your training data
    mfcc_features = preprocess_audio(file_path, max_length)
    
    # Reshape the data for the model input
    mfcc_features = mfcc_features[np.newaxis, :, :]  # Add batch dimension

    # Make prediction
    prediction = model.predict(mfcc_features)
    predicted_index = np.argmax(prediction, axis=1)[0]

    # Map the predicted index back to the corresponding word
    # Load word labels from the original dataset or define the mapping
    word_labels = ["word1", "word2", "word3"]  # Replace with your actual words
    predicted_word = word_labels[predicted_index]

    return predicted_word

# Example usage
file_path = 'cat_test_3.mp3'  # Replace with your audio file path
predicted_word = predict_word(file_path)
print(f'Predicted word: {predicted_word}')
