import os
import numpy as np
from collections import defaultdict

# Path to the folder containing .txt files
folder_path = 'Roy/ML/Resnet_Experiment/txt_storage'

# Initialize a dictionary to accumulate model scores
model_scores = defaultdict(int)
max_score = 0
# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if not file_name.startswith('Difficulty'):
        if file_name == 'Best_overall_models.txt' or file_name == 'A_real_coords.txt' or file_name == 'Best_overall_models_check.txt' or file_name.endswith('Best_check.txt')or file_name.endswith('Best.txt'):
            continue 
        max_score += 3
        file_path = os.path.join(folder_path, file_name)
        #print("Processing file: ", file_path)
        with open(file_path, 'r') as file:
            for line in file:
                print(line)
                models = line.split(':')[1].strip().split(', ')
                
                models = [model for model in models if model]  # remove empty entries
                
                for idx, model in enumerate(models):
                    if not model.startswith('geo'):
                        continue
                    if idx == 0:
                        model_scores[model] += 3
                    elif idx == 1:
                        model_scores[model] += 3
                    elif idx == 2:
                        model_scores[model] += 2
                    elif idx == 3:
                        model_scores[model] += 2
                    elif idx == 4:
                        model_scores[model] += 1
                    elif idx == 5:
                        model_scores[model] += 1
                    elif idx == 6:
                        model_scores[model] += 1
                    elif idx == 7:
                        model_scores[model] += 1    
                    elif idx == 8:
                        model_scores[model] += 1
                    else:
                        model_scores[model] += 1

# Find the 3 highest scores
highest_scores = sorted(model_scores.values(), reverse=True)[:4]
model_scores = dict(sorted(model_scores.items(), key=lambda item: item[1], reverse=True)) 
# save the leaderboard to a file called Best_overall_models_check.txt in the same folder
output_file_path = os.path.join(folder_path, 'Best_overall_models_check.txt')
with open(output_file_path, 'w') as output_file:
    output_file.write("leaderboard = [\n")
    for model, score in model_scores.items():
        output_file.write(f"    ['{model}', {score}],\n")
    output_file.write("]\n")
    
    



avg_batch_size = 0
avg_epoch = 0
avg_error = 0
counter = 0
print(model_scores)

for model, score in model_scores.items():
    if 'check' in model:
        model = model.replace('check_','')
    score= score**0.1
    #disect the model name
    model_names = model.split('_')
    print("Model name parts: ", model_names)
    model_name = model_names[0]+'_' + model_names[1]+'_' + model_names[2]
    epoch = model_names[1].split('e')[0]
    batch = model_names[2].split('b')[0]
    error = float(model_names[3])
    counter +=score
    avg_batch_size += int(batch)*score
    avg_epoch += int(epoch)*score
    avg_error += error*score
if counter > 0:
    avg_batch_size = avg_batch_size/counter
    avg_epoch = avg_epoch/counter
    avg_error = avg_error/counter
    avg_batch_size = int(avg_batch_size)
    avg_epoch = int(avg_epoch)
    avg_error = np.round(avg_error,2)

#construct the average best model name
avg_model_name = model_name + '_' + str(avg_epoch) + 'e_' + str(avg_batch_size) + 'b_' + str(avg_error) + 'k.pth'
print("Average best model name: ", avg_model_name)


print("Regular Models with the highest score out of a possible ", max_score, ":")

print("leaderboard_check = [")
for model, score in model_scores.items():
    if score in highest_scores:
        print(f"    ['{model}', {score}],")
print("]")

