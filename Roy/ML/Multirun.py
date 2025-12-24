import time
tim = time.time()
for i in range(2):
        try:
                with open('Roy/ML/Geoguessrmodel_Trainer_silent.py') as f:
                        exec(f.read())
                        print("------------------------------------------------------------------------------")
        except SystemExit:
            print("SystemExit caught, continuing...")

# Evaluation time
try:
    with open('Roy/ML/Geoguessrmodel_Evaluator_Multimodel_silent.py') as f:
            exec(f.read())
except SystemExit:
    print("SystemExit caught, continuing...")
try:
    with open('Roy/ML/Get_Best_Models.py') as f:
            exec(f.read())
except SystemExit:
    print("SystemExit caught, continuing...")
        
print("Time taken to run the code: ", (time.time() - tim)/60, " minutes")