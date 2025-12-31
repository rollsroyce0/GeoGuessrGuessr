import time
tim = time.time()
for i in range(1):
    with open('Roy/ML/Playground/Playground_Geoguessrmodel_Trainer_Double.py') as f:
        exec(f.read())
        print("------------------------------------------------------------------------------")

# Evaluation time
with open('Roy/ML/Playground/Playground_Geoguessrmodel_Evaluator_Double.py') as f:
        exec(f.read())
with open('Roy/ML/Get_Best_Models.py') as f:
        exec(f.read())
        print("------------------------------------------------------------------------------")
print("Time taken to run the code: ", (time.time() - tim)/60, " minutes")