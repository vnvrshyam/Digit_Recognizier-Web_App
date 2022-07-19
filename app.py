import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import os

UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
model = load_model('model.h5')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('pd.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    im = request.files['image']
    filename = secure_filename(im.filename)
    im.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img_arr = cv2.imread(filename)
    img_arr = cv2.resize(img_arr,(28,28))
    img_arr = cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)
    img_arr = img_arr.reshape((1,784))
    prediction = model.predict(img_arr)
    output = [list(i).index(max(i)) for i in prediction]
    print(output[0])
    return render_template('pd.html', prediction_text='The Given image is a {}'.format(output[0]) )

if __name__ == "__main__":
    app.run(debug=True)