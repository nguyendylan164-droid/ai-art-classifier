from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
import cv2
import numpy as np
from cnn_feature_extraction import extract_cnn_features
from keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
CORS(app)
model = keras.models.load_model("saved_models/ai_art_classifier_resnet.keras")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']

    # read the file directly into memory
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    min_dim = min(height, width)
    start_x = (width // 2) - (min_dim // 2)
    start_y = (height // 2) - (min_dim // 2)
    img_cropped = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # preprocess the image
    img = cv2.resize(img_cropped, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img_rgb, axis=0)

    img_batch_float = np.array(img_batch, dtype=np.float32)
    img_ready = preprocess_input(img_batch_float)

    # make prediction
    prediction = model.predict(img_ready)[0][0]

    if prediction > 0.5:
        result = "Real Art"
    else:
        result = "AI Art"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)