from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("weapon_classifier_model.h5")
IMG_SIZE = 224
class_labels = ['gun', 'knife', 'no_weapon']

def predict_weapon(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    predicted_class = class_labels[np.argmax(pred)]
    return predicted_class, pred[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    img_path = os.path.join("static", img_file.filename)
    img_file.save(img_path)

    pred_class, pred_probs = predict_weapon(img_path)
    return jsonify({'prediction': pred_class, 'image_url': img_path, 'probabilities': pred_probs.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
