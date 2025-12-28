
import os

def read_leaderboard_models(folder_path):
    leaderboard_file = os.path.join(folder_path, 'Best_overall_models_check.txt')
    with open(leaderboard_file, 'r') as f:
        print("Reading leaderboard models from:", leaderboard_file)
        print("File contents:")
        leaderboard_models = []
        for line in f:
            if not(line.__contains__('[') and line.__contains__(']')):
                continue
                
            #print(line.strip())
            leaderboard_models.append(line.replace('[','').replace(']','').replace(',','').replace("'",'').strip().split(' ')[0])
        

        leaderboard_models = [model for model in leaderboard_models if model.endswith('.pth')]
        
    leaderboard_file2 = os.path.join(folder_path, 'Best_overall_models.txt')
    with open(leaderboard_file2, 'r') as f:
        print("Reading additional leaderboard models from:", leaderboard_file2)
        print("File contents:")
        for line in f:
            if not(line.__contains__('[') and line.__contains__(']')):
                continue
                
            #print(line.strip())
            model_name = line.replace('[','').replace(']','').replace(',','').replace("'",'').strip().split(' ')[0]
            if model_name.endswith('.pth') and model_name not in leaderboard_models:
                leaderboard_models.append(model_name)
    return leaderboard_models

if __name__ == "__main__":
    folder_path = 'Roy/Test_Images'
    models = read_leaderboard_models(folder_path)
    print("Leaderboard Models: ", len(models))
    
    for model in os.listdir('Roy/ML/Saved_Models/'):
        if model.endswith('.pth') and model not in models and (not model.__contains__('embedding') or not model.__contains__('Best') or not model.__contains__('low')):
            print("Model not in leaderboard:", model)
            os.rename(os.path.join('Roy/ML/Saved_Models/', model), os.path.join('Roy/ML/Playground/Model_Backup_Bin/', model))
    
    # count models in backup bin
    backup_models = os.listdir('Roy/ML/Playground/Model_Backup_Bin/')
    print("Total models in backup bin:", len(backup_models))
    
    # model files in saved models
    saved_models = os.listdir('Roy/ML/Saved_Models/')
    print("Total models in saved models:", len(saved_models))
    # of which check files
    check_models = [model for model in saved_models if 'check' in model]
    print("Total check models in saved models:", len(check_models))