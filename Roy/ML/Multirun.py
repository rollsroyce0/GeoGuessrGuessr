import time
tim = time.time()
for i in range(3):
        try:
                with open('Roy/ML/Geoguessrmodel_Trainer_silent.py') as f:
                        exec(f.read())
                        print("------------------------------------------------------------------------------")
        except SystemExit:
            print("SystemExit caught, continuing...")

# Evaluation time

    
with open('Roy/ML/Playground/Playground_Geoguessrmodel_Evaluator_Multimodel_silent_Checkpoints.py') as f:
        exec(f.read())

with open('Roy/ML/Geoguessrmodel_Evaluator_Multimodel_silent_check.py') as f:
        exec(f.read())
        


try:
    with open('Roy/ML/Geoguessrmodel_Evaluator_Multimodel_silent.py') as f:
            exec(f.read())
except SystemExit:
    print("SystemExit caught, continuing...")
    
with open('Roy/ML/Playground/Playground_Model_Culling.py') as f:
        exec(f.read())
try:
    with open('Roy/ML/Get_Best_Models.py') as f:
            exec(f.read())
except SystemExit:
    print("SystemExit caught, continuing...")

        
print("Time taken to run the code: ", (time.time() - tim)/60, " minutes")
