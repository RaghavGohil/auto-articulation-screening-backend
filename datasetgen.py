import os
import json
import csv

# Define the folder containing all the subdirectories
data_folder = './All'

# Wordset list
wordset = ['sun', 'beg', 'big', 'box', 'boy', 'bux', 'cat', 'cow', 'cup', 'day', 'dog', 'fan', 'fat', 'fox', 'hat', 
           'hen', 'hot', 'lip', 'man', 'new', 'old', 'out', 'pen', 'rat', 'red', 'run', 'sit', 'wet']

# Define the ranges for locations
rajasthan_range = range(3439, 5302)
uttarpradesh_range = range(1343, 3439)
maharashtra_range = range(1, 1343)

# Determine location based on folder number
def get_location(folder_number):
    folder_number = int(folder_number)
    if folder_number in rajasthan_range:
        return 'Rajasthan'
    elif folder_number in uttarpradesh_range:
        return 'Uttar Pradesh'
    elif folder_number in maharashtra_range:
        return 'Maharashtra'
    return 'Unknown'

# CSV file path
csv_file = 'dataset.csv'

# Open the CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Age', 'Location', 'Audio File', 'Mfcc File' ,'Word', 'Annotation'])
    
    # Iterate through all folders in the data folder
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        
        # Ensure the folder_name is numeric
        if folder_name.isdigit():
            summary_file = os.path.join(folder_path, f'summary{folder_name}.json')
            
            # Check if the summary JSON file exists
            if os.path.exists(summary_file):
                # Open and read the JSON file
                with open(summary_file, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    
                    # Extract details from the JSON file
                    age_group = data.get('ageGroup', 'Unknown')
                    location = get_location(folder_name)
                    
                    # Loop through the sequence list in the JSON
                    for sequence in data.get('sequenceList', []):
                        word = sequence.get('que_text', '').strip().split()[0].lower()  # Take the first word as the main word
                        
                        audio_file = os.path.join(folder_path, sequence.get('recordingName', 'Unknown'))  # Full path to audio file
                        
                        mfcc_file = os.path.join('mfcc_files/'+folder_path, sequence.get('recordingName', 'Unknown')).replace('mp3','csv')  # Full path to mfcc file
                        
                        if not os.path.exists(audio_file):
                            continue

                        # Only process data if isCorrect is True
                        if sequence.get('isCorrect') == True:
                            # Handle noOfMistakes, default to 0 if missing
                            no_of_mistakes = sequence.get('noOfMistakes', 0)
                            
                            # Ensure no_of_mistakes is an integer
                            try:
                                no_of_mistakes = int(no_of_mistakes)
                            except ValueError:
                                no_of_mistakes = 0
                            
                            # Determine annotation based on the number of mistakes
                            annotation = 'Correct' if no_of_mistakes == 0 else 'Incorrect'
                            
                            # Only include words that are in the wordset
                            if word in wordset and annotation == 'Correct':
                                writer.writerow([age_group, location, audio_file, mfcc_file , word, annotation])

print(f"Dataset created successfully and saved as {csv_file}")
