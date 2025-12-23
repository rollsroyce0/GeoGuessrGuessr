import os
import itertools
import pandas as pd

def enumerate_models(folder_path, model_prefix='geo_predictor_nn_', model_suffix='.pth'):
    """
    Enumerate all model files in the specified folder that match the given prefix and suffix.
    
    Parameters:
    folder_path (str): The path to the folder containing model files.
    model_prefix (str): The prefix of the model files to look for.
    model_suffix (str): The suffix of the model files to look for.
    
    Returns:
    list: A list of model file names that match the criteria.
    """
    model_files = []
    model_files_order = []
   
    
    for file_name in os.listdir(folder_path):
        if file_name.startswith(model_prefix) and file_name.endswith(model_suffix):
            model_files.append(file_name)
            
    #order the list according to the table in Best_overall_Models.txt
    for line in open('Roy/Test_Images/Best_overall_models.txt'):
        #print(line)
        model_name = line.strip()
        model_name = model_name.replace('[','').replace(']','').replace(',','').replace("'",'').strip()
        model_name = model_name.split(' ')[0]
        #print(f"Checking for model: {model_name}")
        
        if model_name in model_files:
            model_files_order.append(model_name)
    print(f"Total models found: {len(model_files_order)}")
    
    for model in model_files:
        if model not in model_files_order:
            model_files_order.append(model)
    print(f"Total models after adding missing ones: {len(model_files_order)}")
            
    # save to a dataframe and export as csv
    df = pd.DataFrame(model_files_order, columns=['model_file'])
    df.to_csv(os.path.join('Roy/ML/Second_Level_ML/model_files.csv'), index=False)

    return model_files

enumerate_models('Roy/ML/Saved_Models/')