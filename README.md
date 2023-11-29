# ML Defect Detection System

This project implements a Defect Detection System using machine learning, specifically a Flask web application with TensorFlow for image classification. The system is designed to classify images as either "Defected" or "Non-Defected" based on a pre-trained model.

## Access the Application

The application is hosted on Render and can be accessed through the following link: [Defect Detection System on Railway](https://ml-defect-detection-system-production.up.railway.app/)

## Testing Files

For testing purposes, you can use the following directories containing images:

- **Defected Images:** `archive/casting_512x512_testing/casting_512x512_testing/def_front`
- **Non-Defected Images:** `archive/casting_512x512_testing/casting_512x512_testing/ok_front`

## Usage

1. Access the application using the provided link.
2. Upload an image using the provided form.
3. The system will process the image and classify it as either "Defected" or "Non-Defected."
4. Images not related to the object will be flagged as "Unrelated Image."

## Note on Unrelated Images

Please be aware that the system is configured to reject images not related to the object. This ensures that only relevant images are processed for defect classification.
