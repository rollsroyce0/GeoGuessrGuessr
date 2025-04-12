leaderboard =[["geo_predictor_nn_500e_64b_926k", 14987],
["geo_predictor_nn_500e_64b_952k", 16037], 
["geo_predictor_nn_452e_64b_914k", 11209], 
["geo_predictor_nn_500e_64b_953k", 10024], 
#------ Below is placeholder for future models ------
["geo_predictor_nn_500e_8192b_7551k", 605], 
["geo_predictor_nn_500e_64b_926k", 0], 
["geo_predictor_nn_500e_64b_926k", 0], 
["geo_predictor_nn_500e_64b_926k", 0], 
["geo_predictor_nn_500e_64b_926k", 0], 
["geo_predictor_nn_500e_64b_926k", 0]]

# Sort the leaderboard by points in descending order
leaderboard.sort(key=lambda x: x[1], reverse=True)
print("Leaderboard:")
for i, (model_name, points) in enumerate(leaderboard):
    print(f"{i + 1}. {model_name}: {points} points")
    