import time
tim = time.time()
for i in range(3):
    with open('Roy/ML/Geoguessrmodel_Trainer_silent.py') as f:
        exec(f.read())
        print("------------------------------------------------------------------------------")

# Evaluation time
with open('Roy/ML/Playground/Playground_Geoguessrmodel_Evaluator_Multimodel_silent.py') as f:
        exec(f.read())
with open('Roy/ML/Get_Best_Models.py') as f:
        exec(f.read())
        
print("Time taken to run the code: ", (time.time() - tim)/60, " minutes")