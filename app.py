from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Get the absolute path to the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'defect_classification_model.h5')

defect_classification_model = load_model(model_path)


# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define route for handling image upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    # Access the uploaded image from the request
    uploaded_file = request.files['file']

    # Perform validation to reject unrelated images
    if uploaded_file and allowed_file(uploaded_file.filename):
        # Process the image and call the deployed model for classification
        result, confidence = classify_image(uploaded_file)
        if confidence == 0:  # Check if the image was rejected
            return render_template('error.html', message=result)
        # Convert confidence to a percentage for display
        confidence_percent = "{:.2f}%".format(confidence * 100)
        return render_template('result.html', result=result, accuracy=confidence_percent)
    else:
        return render_template('error.html', message='Invalid file format or unrelated image')


# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


# Function to check if the image is grayscale
def is_grayscale(img_array):
    if np.array_equal(img_array[..., 0], img_array[..., 1]) and np.array_equal(img_array[..., 1], img_array[..., 2]):
        return True
    return False


# Function to call the deployed model for classification
def classify_image(uploaded_file):
    # Open the image using PIL (Python Imaging Library)
    img = Image.open(uploaded_file.stream).convert("RGB")

    # Check if the image is grayscale
    if not is_grayscale(np.array(img)):
        return 'Unrelated image', 0

    img = img.resize((128, 128))  # Resize the image to match the model's expected sizing
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Call the model for predictions
    predictions = defect_classification_model.predict(img_array)

    # Extract the confidence of the prediction
    confidence = predictions[0][0]

    # Convert predictions to class labels
    predicted_class = "Non-Defective" if confidence >= 0.5 else "Defective"

    # Return both class label and confidence
    return predicted_class, confidence


if __name__ == '__main__':
    app.run(debug=True)
