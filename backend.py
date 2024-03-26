from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import load_model
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)
model = load_model('Pneumonia_ResNet50.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the POST request
    file = request.files['file']

    # Read the image file and preprocess it
    img = Image.open(file).convert('RGB')
    img = img.resize((224, 224))

    # Convert image to numpy array
    img_array = np.array(img)

    # Add a batch dimension so that the shape of the image is (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Decode prediction
    if prediction[0][0] > 0.5:
        result = "Normal"
        accuracy = prediction[0][0] * 100
    else:
        result = "Pneumonia"
        accuracy = (1 - prediction[0][0]) * 100

    # Convert image to base64 string
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Prepare response
    response = {
        'class': result,
        'accuracy': accuracy,
        'imageData': img_str
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)