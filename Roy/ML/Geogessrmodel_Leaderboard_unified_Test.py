# Import leaderboard data from a file
leaderboard = []
with open('Roy/ML/leaderboard.txt', 'r') as file:
    for line in file:
        # Extract the model name and points from each line
        parts = line.replace("'", "").replace("[", "").replace("]", "")
        parts = parts.split(",")
        parts = [x.strip() for x in parts]
        if len(parts) != 3: # Three since one is after the last comma (empty string)
            continue
        model_name = parts[0]
        points = int(parts[1])
        leaderboard.append([model_name, points])
        

print(leaderboard)

# Human Leaderboard
Human_scores = [
    ['Matteo', 15786],
    ['Enrico', 13086],
    ['Luca', 10475],
    ['Death', 10036],
    ['Nic', 10171],
    ['Vikram', 12800],
    ['Sam', 10728],
    ['Dan', 11155],
]

# Remove Duplicates from the leaderboard list
leaderboard = leaderboard[:1] + [x for x in leaderboard[1:] if x not in leaderboard[:1]]

# Sort the leaderboard by points in descending order
leaderboard.sort(key=lambda x: x[1], reverse=True)
print("Leaderboard:")
for i, (model_name, points) in enumerate(leaderboard):
    print(f"{i + 1}. {model_name}: {points} points")

Human_scores.sort(key=lambda x: x[1], reverse=True)
print("Human Leaderboard:")
for i, (name, points) in enumerate(Human_scores):
    print(f"{i + 1}. {name}: {points} points")
    
# Human or bot combined leaderboard
combined_leaderboard = leaderboard + Human_scores
combined_leaderboard.sort(key=lambda x: x[1], reverse=True)
print("Combined Leaderboard:")
for i, (name, points) in enumerate(combined_leaderboard):
    print(f"{i + 1}. {name}: {points} points")

    