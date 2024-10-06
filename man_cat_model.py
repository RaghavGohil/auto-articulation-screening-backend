import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
df = pd.read_csv("man_cat_dataset.csv")
print(len(df))

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

# Function to pad MFCC arrays to a fixed size (e.g., the maximum length of any MFCC array)
def pad_mfcc(mfcc, max_length):
    if mfcc.shape[0] < max_length:
        # Pad with zeros if MFCC is shorter than max_length
        return np.pad(mfcc, ((0, max_length - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        # Truncate if longer than max_length
        return mfcc[:max_length, :]

# Find the maximum length of any MFCC array
max_mfcc_length = max([mfcc.shape[0] for mfcc in df['MFCC']])

# Pad all MFCC arrays to the same length
df['Padded_MFCC'] = df['MFCC'].apply(lambda x: pad_mfcc(x, max_mfcc_length))

# Convert MFCCs into feature vectors and concatenate with one-hot encoded categorical features
X_mfcc = np.stack(df['Padded_MFCC'].values)  # Stacking padded MFCC arrays
X_metadata = np.array(encoded_features)  # Metadata features
X_metadata = np.repeat(X_metadata[:, np.newaxis, :], X_mfcc.shape[1], axis=1)  # Repeat metadata for concatenation

# Concatenate MFCC and metadata features along the last dimension
X = np.concatenate([X_mfcc, X_metadata], axis=2)

# Correct Target Variable (y)
# Assuming 'Label' column contains 'cat' and 'man', convert them to integers 0 for cat, 1 for man.
df['Word'] = df['Word'].map({'cat': 0, 'man': 1})

# One-hot encoding of target labels for binary classification (each label is independent)
y = tf.keras.utils.to_categorical(df['Word'], num_classes=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple deep learning model (with LSTM)
model = models.Sequential([
    layers.Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])),  # Use Masking for variable lengths
    layers.LSTM(128, return_sequences=False),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='sigmoid')  # Sigmoid for independent binary outputs
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
# Compile the model with binary crossentropy

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model
try:
    model.save('man_cat_model.keras')
    print('Saved model successfully!')
except:
    print('Couldn\'t save model...')
