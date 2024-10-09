import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models

# Predefine the encoder categories for age and location
encoder = OneHotEncoder(sparse_output=False, categories=[['less than 5', '5-7', '7-10', '10-12 years', '12-15 years', '15-18'],
                                                         ['Rajasthan', 'Maharashtra', 'Uttar Pradesh']])
# Pre-fit the encoder with all possible combinations
encoder.fit([['10-12 years', 'Rajasthan']])  # Dummy fit with valid data; will only use transform later

# Function to find the maximum MFCC length and unique words in the dataset
def find_max_mfcc_length_and_words(dataset_csv):
    max_length = 0
    unique_words = set()  # Use a set to store unique words
    
    # Load the dataset
    try:
        data = pd.read_csv(dataset_csv)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return None, None
    
    # Iterate through each MFCC file path in the dataset
    for _, row in data.iterrows():
        mfcc_file = row['Mfcc File']
        word = row['Word']  # Assuming there's a 'Word' column
        
        # Add the word to the set of unique words
        unique_words.add(word)
        
        # Check if the file exists
        if os.path.exists(mfcc_file):
            try:
                # Load MFCC data
                mfcc_data = pd.read_csv(mfcc_file).values
                mfcc_length = mfcc_data.shape[0]  # Get the number of time steps
                
                # Update max_length if current is greater
                if mfcc_length > max_length:
                    max_length = mfcc_length
            
            except Exception as e:
                print(f"Error loading {mfcc_file}: {e}")
        else:
            print(f"MFCC file not found: {mfcc_file}")
    
    return max_length, list(unique_words)  # Return max_length and unique words as a list

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
def predict_word(mfcc_file_path, age, location, expected_word, threshold=0.8, max_mfcc_length=None, word_labels=None):
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
    model = tf.keras.models.load_model('cnn_word_prediction_model.keras')

    # Reshape the input to match model's input shape (batch size, time steps, features)
    input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension
    input_features = np.expand_dims(input_features, axis=-1)  # Add channel dimension

    # Predict the word and get the confidence score
    prediction = model.predict(input_features)
    confidence = prediction[0]  # Array of size [num_classes]

    # Convert softmax outputs into a label
    predicted_label = np.argmax(confidence)  # Index of highest confidence
    predicted_confidence = confidence[predicted_label]  # Confidence for the predicted class

    # Map predicted label index to the actual word
    predicted_word = word_labels[predicted_label]

    # Print prediction details
    print(f"Predicted word: '{predicted_word}' with confidence: {predicted_confidence:.2f}")

    # Compare the confidence score with the threshold to detect misarticulation
    if predicted_word == expected_word:
        if predicted_confidence >= threshold:
            print(f"Correct articulation! Predicted word: '{predicted_word}', Confidence: {predicted_confidence:.2f}")
            return 'Correct', predicted_confidence
        else:
            print(f"Word '{expected_word}' was misarticulated with low confidence: {predicted_confidence:.2f}")
            return 'Misarticulated', predicted_confidence
    else:
        print(f"Misarticulated! Expected: '{expected_word}', Predicted: '{predicted_word}', Confidence: {predicted_confidence:.2f}")
        return 'Misarticulated', predicted_confidence

if __name__ == "__main__":
    # Set the path to your dataset CSV file
    dataset_csv = 'balanced_dataset.csv'  # Update this path
    
    # Find the maximum MFCC length and unique words from the dataset
    max_mfcc_length, word_labels = find_max_mfcc_length_and_words(dataset_csv)
    print(f"Maximum MFCC length found: {max_mfcc_length}")
    print(f"Unique words found in dataset: {word_labels}")

    # Specify the MFCC file, age, location, and expected word directly
    mfcc_file_path = 'mfcc_files/./All/3276/HI_S1_W_1.csv'  # Path to MFCC file
    age = '7-10'  # Age group
    location = 'Maharashtra'  # Location
    expected_word = 'cat'  # Expected word spoken in the audio (e.g., 'cat' or 'man')

    # Set a threshold for detecting misarticulation (e.g., 80% confidence)
    threshold = 0.8

    # Call the prediction function with misarticulation detection
    result, confidence = predict_word(mfcc_file_path, age, location, expected_word, threshold, max_mfcc_length=max_mfcc_length, word_labels=word_labels)
