import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dataset
df = pd.read_csv('balanced_dataset.csv')

# Function to load MFCC data and skip missing files
def load_mfcc(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path).values
    else:
        return None  # Return None for missing files

# Apply the function and filter out rows with missing MFCC files
df['MFCC'] = df['Mfcc File'].apply(load_mfcc)
df = df[df['MFCC'].notna()]  # Remove rows with missing MFCCs

# One-hot encoding for Age and Location
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['Age', 'Location']])

# Function to pad MFCC arrays to a fixed size
def pad_mfcc(mfcc, max_length):
    if mfcc.shape[0] < max_length:
        return np.pad(mfcc, ((0, max_length - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        return mfcc[:max_length, :]

# Find the maximum length of any MFCC array
max_mfcc_length = max([mfcc.shape[0] for mfcc in df['MFCC']])

# Pad all MFCC arrays to the same length
df['Padded_MFCC'] = df['MFCC'].apply(lambda x: pad_mfcc(x, max_mfcc_length))

# Prepare features
X_mfcc = np.stack(df['Padded_MFCC'].values)  # Stacking padded MFCC arrays
X_metadata = np.array(encoded_features)  # Metadata features
X_metadata = np.repeat(X_metadata[:, np.newaxis, :], X_mfcc.shape[1], axis=1)  # Repeat metadata for concatenation
X = np.concatenate([X_mfcc, X_metadata], axis=2)

# Target variable - map words to integers
word_labels = df['Word'].unique()  # Assuming these are the 26 words you have
word_to_index = {word: idx for idx, word in enumerate(word_labels)}  # Create a mapping

df['Word'] = df['Word'].map(word_to_index)  # Map words to indices
y = tf.keras.utils.to_categorical(df['Word'], num_classes=len(word_labels))  # One-hot encode labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input for CNN (add channel dimension)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Define the CNN model
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(word_labels), activation='softmax')  # Output layer matches number of words
])

# Compile the model
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture training history
history_cnn = model_cnn.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test))

# Evaluate the model
test_loss, test_acc = model_cnn.evaluate(X_test_cnn, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model
try:
    model_cnn.save('cnn_word_prediction_model.keras')
    print('Saved CNN model successfully!')
except:
    print('Couldn\'t save CNN model...')

# Plotting training & validation accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history_cnn.png')
    plt.show()

# Call the function to plot and save graphs
plot_training_history(history_cnn)

print("Graphs saved as 'training_history_cnn.png'")
