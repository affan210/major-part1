import sys
import os
import cv2
import numpy as np

# Keras
import keras
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C:/Users/KIIT/Downloads/Major/model.h5'

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()  #Necessary for Prediction Result
print('Model loading... Serving...!')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    test = []
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    test.append(image)
    test = np.array(test) / 255.0

    res = model.predict(test)
    r = np.argmax(res, axis=1)
    if r == 0:
        return ("Glioma Tumor Detected")
    elif r==1:
        return ("Meningioma Tumor Detected")
    elif r==2:
        return ("No Tumor Detected")
    elif r==3:
        return ("Pituitary Tumor Detected")
    return None


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        return preds
    return None


if __name__ == '__main__':
    app.run()
