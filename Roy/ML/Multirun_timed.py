import time 
filepath = 'Roy/ML/Geoguessrmodel_Trainer_silent.py'

print("Starting the script...")
print("How much time should the code run for?")
timelmt = input("Enter the time limit in the following format 1h15m30s: ")

# if any of the h/m/s are missing, add them as 0
if 'h' not in timelmt:
    timelmt = '0h' + timelmt
if 'm' not in timelmt:
    timelmt = timelmt.split('h')[0] + '0m' + timelmt.split('h')[1]
if 's' not in timelmt:
    timelmt = timelmt + '0s'

timelmt = time.strptime(timelmt, '%Hh%Mm%Ss')
print("Time limit in seconds: ", timelmt.tm_hour * 3600 + timelmt.tm_min * 60 + timelmt.tm_sec)
timelmt = timelmt.tm_hour * 3600 + timelmt.tm_min * 60 + timelmt.tm_sec
print("Time limit in seconds: ", timelmt)
#quit()
# run the code once and time it, figure out how long it takes to run
start_time = time.time()
with open(filepath) as f:
        exec(f.read())
end_time = time.time()
print("Time taken to run the code once: ", (end_time - start_time)/60, " minutes")
total_runs_possible = int(timelmt / (end_time - start_time))
print("Total runs possible: ", total_runs_possible)

# run the code in a loop for the time limit
for i in range(total_runs_possible):
    start_time = time.time()
    with open(filepath) as f:
        exec(f.read())
    end_time = time.time()
    print("Time taken to run the code: ", (end_time - start_time)/60, " minutes")
    print("Total runs possible: ", total_runs_possible - i - 1)
    
    # Check if the time limit has been reached
    if end_time - start_time > timelmt:
        print("Time limit reached. Stopping the script after ", i, " runs.")
        break
    print("------------------------------------------------------------------------------")

# Evaluation time
with open('Roy/ML/Playground/Playground_Geoguessrmodel_Evaluator_Multimodel_silent.py') as f:
        exec(f.read())
with open('Roy/ML/Get_Best_Models.py') as f:
        exec(f.read())