from flask import Flask, render_template, request
from model import load_model, predict
import os

app = Flask(__name__)
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join('images', file.filename)
        file.save(filepath)
        prediction = predict(filepath, model)
        return prediction

if __name__ == '__main__':
    app.run(debug=True)
