import time
import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ipsc_predictor import run_models

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'txt', 'tsv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    result_filename = None
    bar_img = None
    jitter_img = None
    bar_file = None
    jitter_file = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_filename = secure_filename(file.filename)
            saved_filename = f"{timestamp}_{base_filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(file_path)

            # Run the model and generate plots
            result_dict, bar_img, jitter_img, bar_file, jitter_file = run_models(file_path)
            result_df = pd.DataFrame.from_dict(result_dict, orient='index')

            # Save result to Excel
            result_filename = f"{timestamp}_result.xlsx"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            result_df.to_excel(result_path)

            # Return only preview (head) in HTML
            result = result_df.head().round(4).to_dict(orient='index')

    return render_template(
        'index.html',
        result=result,
        result_file=result_filename,
        bar_img=bar_img,
        jitter_img=jitter_img,
        bar_file=bar_file,
        jitter_file=jitter_file
    )


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)

    app.run(debug=True)
