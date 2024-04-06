from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.resnet import ResNet50
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load the pre-trained ResNet50 model without the top classification layer
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

# Define the model architecture using functional API
input_tensor = Input(shape=(224, 224, 3))
x = base_model(input_tensor)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output_tensor = Dense(2, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

# Load the trained weights
model.load_weights('Pneumonia_VGG16.weights.h5')

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

    # Print the predicted probabilities for each class DEBUG
    print("Predicted probabilities:", prediction)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction[0])

    # Define the class labels
    class_labels = ['NORMAL', 'PNEUMONIA']

    # Get the predicted class label and accuracy
    predicted_class = class_labels[predicted_class_index]
    accuracy = prediction[0][predicted_class_index] * 100

    # Convert image to base64 string
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Prepare response
    response = {
        'class': predicted_class,
        'accuracy': accuracy,
        'imageData': img_str
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5528)