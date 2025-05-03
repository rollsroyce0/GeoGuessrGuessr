import os
from collections import defaultdict

# Path to the folder containing .txt files
folder_path = 'Roy/Test_Images'

# Initialize a dictionary to accumulate model scores
model_scores = defaultdict(int)

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt') and not file_name.startswith('Difficulty'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            for line in file:
                models = line.split(':')[1].strip().split(', ')
                models = [model for model in models if model]  # remove empty entries
                
                for idx, model in enumerate(models):
                    if idx == 0:
                        model_scores[model] += 6
                    elif idx == 1:
                        model_scores[model] += 5
                    elif idx == 2:
                        model_scores[model] += 5
                    elif idx == 3:
                        model_scores[model] += 3
                    elif idx == 4:
                        model_scores[model] += 2
                    else:
                        model_scores[model] += 1

# Find the 3 highest scores
highest_scores = sorted(model_scores.values(), reverse=True)[:3]
print("Top 3 scores:")
print(highest_scores)

print("Models with the highest score:")
print("leaderboard = [")
for model, score in model_scores.items():
    if score in highest_scores:
        print(f"    ['{model}', {score}],")
print("]")
