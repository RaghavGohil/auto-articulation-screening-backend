import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import joblib
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.preprocessing._encoders')

def preprocess_location(location, location_encoder):
    return location_encoder.transform(np.array([location]).reshape(-1, 1))[0]

def preprocess_age(age, age_encoder):
    return age_encoder.transform(np.array([age]).reshape(-1, 1))[0]

def process_batch(data_batch, location_encoder, age_encoder, max_len):
    features = []
    labels = []
    for _, row in data_batch.iterrows():
        mfcc_file = row['Mfcc File']
        label = row['Word']
        location = row['Location']
        age = row['Age']
        
        if pd.notna(mfcc_file) and os.path.exists(mfcc_file):
            try:
                # Load MFCC features and flatten them
                feature = pd.read_csv(mfcc_file).values.flatten()
                feature = feature.astype(np.float32)
                
                # Pad or truncate the feature vector to ensure consistent length
                if len(feature) < max_len:
                    feature = np.pad(feature, (0, max_len - len(feature)), mode='constant')
                else:
                    feature = feature[:max_len]
                
                # One-hot encode location and age
                location_encoded = preprocess_location(location, location_encoder)
                age_encoded = preprocess_age(age, age_encoder)
                
                # Concatenate all features into one array
                combined_feature = np.hstack([feature, location_encoded, age_encoded])
                features.append(combined_feature)
                labels.append(label)
            except Exception as e:
                print(f'Error processing file {mfcc_file}: {e}')
    
    return np.array(features), np.array(labels)

def load_data_in_batches(csv_file, location_encoder, age_encoder, max_len, batch_size=1000):
    data = pd.read_csv(csv_file)
    num_batches = (len(data) + batch_size - 1) // batch_size
    all_features = []
    all_labels = []
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch_data = data[i*batch_size:(i+1)*batch_size]
        features, labels = process_batch(batch_data, location_encoder, age_encoder, max_len)
        all_features.append(features)
        all_labels.append(labels)
    
    return np.vstack(all_features), np.hstack(all_labels)

# Initialize encoders
location_encoder = OneHotEncoder()
age_encoder = OneHotEncoder()

# Sample data for fitting encoders
sample_data = pd.read_csv('cat_dataset.csv')
location_encoder.fit(sample_data[['Location']])
age_encoder.fit(sample_data[['Age']])

# Determine the maximum length of MFCC features
sample_features = [pd.read_csv(f).values.flatten().astype(np.float32) for f in sample_data['Mfcc File'].dropna() if os.path.exists(f)]
max_len = max(len(f) for f in sample_features) if sample_features else 5000  # Fallback length if no features

# Load data
features, labels = load_data_in_batches('cat_dataset.csv', location_encoder, age_encoder, max_len)

# Dimensionality Reduction: Reduce feature size using PCA
n_components = 1000  # Adjust based on memory constraints
pca = PCA(n_components=n_components)
features_reduced = pca.fit_transform(features)
print(f'Reduced feature shape: {features_reduced.shape}')

# Normalize features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_reduced)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features_normalized, encoded_labels, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
n_estimators = 100  # Number of trees in the forest
random_forest_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions
y_pred = random_forest_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'F1 Score: {f1:.4f}')

# Save the model if needed
joblib.dump(random_forest_model, 'random_forest_model.pkl')
print('Model saved to random_forest_model.pkl')

