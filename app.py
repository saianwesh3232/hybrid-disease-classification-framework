import os
import numpy as np
import tensorflow as tf
import cv2 
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# =========================
# CONFIG
# =========================
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model("model/Multi_Class_Disease_Detection_System.h5")

# Apne dataset ke labels
class_names = ['dog_dental_disease','dog_dermatitis','dog_healthy','dog_demodicosis','dog_ringworm','dog_Skin_Disease','human_glioma','human_meningioma','human_notumor','human_pituitary']


# =========================
# HOME ROUTE
# =========================
@app.route('/')
def home():
    return render_template('index.html')

# =========================
# PREDICTION ROUTE
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # =========================
    # IMAGE PREPROCESSING
    # =========================
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # =========================
    # PREDICTION
    # =========================
    preds = model.predict(img_array)

    predicted_class = class_names[np.argmax(preds)]
    confidence = round(np.max(preds) * 100, 2)


    # =========================
    # RETURN RESULT
    # =========================
    return render_template(
        'index.html',
        prediction=predicted_class,
        confidence=confidence,
        img_path=filepath
    )

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True) 
