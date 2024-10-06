import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

# Predefine the encoder categories for age and location
encoder = OneHotEncoder(sparse_output=False, categories=[['less than 5', '5-7', '7-10', '10-12 years', '12-15 years', '15-18'],
                                                         ['Rajasthan', 'Maharashtra', 'Uttar Pradesh']])
# Pre-fit the encoder with all possible combinations
encoder.fit([['10-12 years', 'Rajasthan']])  # Dummy fit with valid data; will only use transform later

# Function to load MFCC data from a given file path
def load_mfcc_for_prediction(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path).values
    else:
        raise FileNotFoundError(f"MFCC file not found at {file_path}")

# Function to pad MFCC arrays to the specified maximum length
def pad_mfcc(mfcc, max_length):
    if mfcc.shape[0] < max_length:
        return np.pad(mfcc, ((0, max_length - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        return mfcc[:max_length, :]

# Function to load the model and make predictions with confidence
def predict_word(mfcc_file_path, age, location, expected_word, threshold=0.8, max_mfcc_length=50):
    # Load MFCC file
    try:
        mfcc = load_mfcc_for_prediction(mfcc_file_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Pad MFCC array to match the max_mfcc_length used during training
    padded_mfcc = pad_mfcc(mfcc, max_mfcc_length)

    # One-hot encoding for age and location (using pre-fitted encoder)
    encoded_metadata = encoder.transform([[age, location]])

    # Repeat the metadata to match the MFCC's time steps
    metadata_repeated = np.repeat(encoded_metadata[:, np.newaxis, :], max_mfcc_length, axis=1)

    # Concatenate MFCC and metadata along the feature axis (time steps × (mfcc_features + metadata_features))
    input_features = np.concatenate([padded_mfcc, metadata_repeated[0]], axis=1)

    # Load the trained model
    model = tf.keras.models.load_model('man_cat_model.keras')

    # Reshape the input to match model's input shape (batch size, time steps, features)
    input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

    # Predict the word ('cat' or 'man') and get the confidence score
    prediction = model.predict(input_features)
    confidence = prediction[0]  # Array of size 2 [confidence_man, confidence_cat]

    # Convert softmax outputs into a label
    predicted_label = 'man' if confidence[0] > confidence[1] else 'cat'
    predicted_confidence = max(confidence)  # Confidence for the predicted class

    # Print prediction details
    print(f"Predicted word: '{predicted_label}' with confidence: {predicted_confidence:.2f}")

    # Compare the confidence score with the threshold to detect misarticulation
    if predicted_label == expected_word:
        if predicted_confidence >= threshold:
            print(f"Correct articulation! Predicted word: '{predicted_label}', Confidence: {predicted_confidence:.2f}")
            return 'Correct', predicted_confidence
        else:
            print(f"Word '{expected_word}' was misarticulated with low confidence: {predicted_confidence:.2f}")
            return 'Misarticulated', predicted_confidence
    else:
        print(f"Misarticulated! Expected: '{expected_word}', Predicted: '{predicted_label}', Confidence: {predicted_confidence:.2f}")
        return 'Misarticulated', predicted_confidence

if __name__ == "__main__":
    # You can edit this path and metadata for making predictions
    mfcc_file_path = 'cat_mfcc_3'  # Path to MFCC file
    age = '5-7'  # Age group (can be '10-12 years', '12-15 years', etc.)
    location = 'Maharashtra'  # Location (can be 'Rajasthan', 'Maharashtra', 'Uttar Pradesh')
    expected_word = 'cat'  # Expected word spoken in the audio (e.g., 'cat' or 'man')

    # Set a threshold for detecting misarticulation (e.g., 80% confidence)
    threshold = 0.8

    # Call the prediction function with misarticulation detection
    result, confidence = predict_word(mfcc_file_path, age, location, expected_word, threshold, max_mfcc_length=50)

