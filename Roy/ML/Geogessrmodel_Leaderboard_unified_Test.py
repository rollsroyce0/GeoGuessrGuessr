leaderboard = [
    ['geo_predictor_nn_452e_64b_914k.pth', 11209],
    ['geo_predictor_nn_500e_1024b_1412k.pth', 11423],
    ['geo_predictor_nn_500e_256b_899k.pth', 12234],
    ['geo_predictor_nn_500e_64b_926k.pth', 14987],
    ['geo_predictor_nn_500e_64b_952k.pth', 16037],
    ['geo_predictor_nn_500e_64b_953k.pth', 10024],
    ['geo_predictor_nn_500e_8192b_7551k.pth', 605],
]

# Remove Duplicates from the leaderboard
leaderboard = list(dict.fromkeys(leaderboard))

# Sort the leaderboard by points in descending order
leaderboard.sort(key=lambda x: x[1], reverse=True)
print("Leaderboard:")
for i, (model_name, points) in enumerate(leaderboard):
    print(f"{i + 1}. {model_name}: {points} points")
    