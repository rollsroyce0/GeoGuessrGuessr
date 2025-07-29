import os
from flask import Flask, render_template, request
from Geoguessrmodel_Evaluator_Multimodel_silent import main as run_inference

app = Flask(__name__)
list_of_maps = ['Game', 'Validation', 'Super', 'Verification', 'Ultra', 'Extreme', 'Chrome',
                'World', 'Task', 'Enlarged', 'Exam', 'Google', 'Zurich', 'Friends', 'Full',
                'Entire', 'Moscow']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        testtype = request.form['testtype']
        # run inference and return coords
        # After you load images and paths in run_inference, collect the image file names
        final, highest, difficulty, avg_scores, median_scores, pred_coords, real_coords, errors, points, img_paths = run_inference(testtype)

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
