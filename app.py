from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PredictCharacters import *

app = Flask(__name__)

# Thiết lập thư mục lưu trữ ảnh và cho phép các định dạng phù hợp
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_filename = 'models/finalized_model_rbf_1.sav'
svc_model = pickle.load(open(model_filename, 'rb'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(os.path.abspath(filepath))
        
        prediction = recognize_license_plate(img, svc_model)
        
        return render_template('index.html', filename=filename, prediction=prediction)
    else:
        return render_template('index.html', error='Invalid file format')
if __name__ == '__main__':
    app.run(debug=True)