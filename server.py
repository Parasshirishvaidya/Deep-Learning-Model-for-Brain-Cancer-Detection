from flask import Flask, request, render_template
import base64
from PIL import Image
import io
import pickle
app = Flask(__name__)

with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'image' not in request.files:
        return 'No file uploaded'

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # call your prediction model function
    result = model.predict(img)[0][0]

    return result
