from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

weights_path = '../vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = VGG16(weights=weights_path, include_top=False, input_shape=(224, 224, 3))
# Freeze base model layers
base_model.trainable = False

# Define the model architecture
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

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