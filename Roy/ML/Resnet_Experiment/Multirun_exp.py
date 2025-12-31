import shutil
import time
tim = time.time()
for i in range(10):
    with open('Roy/ML/Resnet_Experiment/Geoguessrmodel_Trainer_silent_resnet_exp_experimental.py') as f:
        exec(f.read())
        print("------------------------------------------------------------------------------")

# Evaluation time
with open('Roy/ML/Resnet_Experiment/Geoguessrmodel_Evaluator_Multimodel_silent_resnet_exp.py') as f:
        exec(f.read())
with open('Roy/ML/Resnet_Experiment/Get_Best_Models_exp.py') as f:
        exec(f.read())
        
del_checkpoint_models = input("Delete checkpoint models? (y/n): ")
if del_checkpoint_models.lower() == 'y':
    import os
    checkpoint_model_folder = 'Roy/ML/Resnet_Experiment/Saved_Models_New/Checkpoint_Models_NN'
    for filename in os.listdir(checkpoint_model_folder):
        file_path = os.path.join(checkpoint_model_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f'Deleted file: {file_path}')
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f'Deleted directory: {file_path}')
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
        
print("Time taken to run the code: ", (time.time() - tim)/60, " minutes")