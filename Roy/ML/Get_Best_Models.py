import os
from collections import Counter

# Path to the folder containing .txt files
folder_path = 'Roy/Test_Images'

# Initialize a Counter to count model occurrences
model_counter = Counter()

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt') and not file_name.startswith('Difficulty'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            # Read each line and update the counter
            unique_models = set()
            for line in file:
                # Extract models from the line
                models = line.split(':')[1].strip().split(', ')
                # Add models to the set to remove duplicates within the file
                # remove the : entry from the list of models
                models = [model for model in models if model != '']
                
                unique_models.update(models)
            # Update the counter with unique models
            for model in unique_models:
                model_counter[model] += 1
# Print the count of each model
highest_count = max(model_counter.values())

print("Models with the highest count:")
for model, count in model_counter.items():
    if count == highest_count:
        print(f"{model}: {count} times")