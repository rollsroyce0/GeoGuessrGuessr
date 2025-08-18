import os
from flask import Flask, render_template, request
from Geoguessrmodel_Evaluator_Multimodel_silent import main as run_inference

app = Flask(__name__)
list_of_maps = ['All', 'Game', 'Validation', 'Super', 'Verification', 'Ultra', 'Extreme', 'Chrome',
                'World', 'Task', 'Enlarged', 'Exam', 'Google', 'Zurich', 'Friends', 'Full',
                'Entire', 'Moscow', 'Berne', 'Beans', 'Geneva']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        testtype = request.form['testtype']
        if testtype == 'All':
            final_temp, highest_temp, difficulty_temp, avg_scores_temp, median_scores_temp, pred_coords_temp, real_coords_temp, errors_temp, points_temp, img_paths_temp = [], [], [], [], [], [], [], [], [], []
            final, highest, difficulty, avg_scores, median_scores, pred_coords, real_coords, errors, points, img_paths = [], [], [], [], [], [], [], [], [], []
            for map_type in list_of_maps:
                if map_type != 'All':
                    # run inference for each map type
                    final_temp, highest_temp, difficulty_temp, avg_scores_temp, median_scores_temp, pred_coords_temp, real_coords_temp, errors_temp, points_temp, img_paths_temp = run_inference(map_type)
                    final.append(final_temp)
                    highest.append(highest_temp)
                    difficulty.append(difficulty_temp)
                    avg_scores.append(avg_scores_temp)
                    median_scores.append(median_scores_temp)
                    pred_coords.extend(pred_coords_temp)
                    real_coords.extend(real_coords_temp)
                    errors.extend(errors_temp)
                    points.extend(points_temp)
                    img_paths.extend(img_paths_temp)
        else:
            # run inference and return coords
            # After you load images and paths in run_inference, collect the image file names
            final, highest, difficulty, avg_scores, median_scores, pred_coords, real_coords, errors, points, img_paths = run_inference(testtype)

        print("Final:", final)
        print("Highest:", highest)
        print("Difficulty:", difficulty)
        print("Average Scores:", avg_scores)
        print("Median Scores:", median_scores)
        print("Predicted Coords:", pred_coords)
        print("Real Coords:", real_coords)
        print("Errors:", errors)
        print("Points:", points)
        print("Image Paths:", img_paths)

        # Convert everything
        real_coords = [list(coord) for coord in real_coords]
        pred_coords = [list(coord) for coord in pred_coords]
        errors = list(errors)
        points = list(points)
        image_names = [os.path.basename(path) for path in img_paths] # get them from static
        print("Image names:", image_names)
        


        return render_template("results.html",
                               testtype=testtype,
                               final_total=final,
                               highest_total=highest,
                               difficulty_avg=difficulty,
                               real_coords=real_coords,
                               pred_coords=pred_coords,
                               errors=errors,
                               points=points,
                               image_names=image_names)
    return render_template("index.html", testtypes=list_of_maps)

if __name__ == "__main__":
    app.run(debug=True) 
