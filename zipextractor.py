import os
import zipfile

# Define the path to the folder containing ZIP files and the destination directory
zip_folder = input('Enter zip folder.')
extract_folder = input('Enter extract folder.')

# Create the extract folder if it doesn't exist
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

# Loop through all files in the ZIP folder
for filename in os.listdir(zip_folder):
    if filename.endswith('.zip'):
        # Construct full file path
        file_path = os.path.join(zip_folder, filename)
        
        # Create a folder in the destination for this specific ZIP file
        folder_name = os.path.splitext(filename)[0]  # Remove the .zip extension
        extract_path = os.path.join(extract_folder, folder_name)
        
        # Create the specific folder for extraction if it doesn't exist
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        
        # Extract the ZIP file into its corresponding folder
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f'Extracted: {filename} to {extract_path}')

print('All zip files have been extracted into their respective folders.')

