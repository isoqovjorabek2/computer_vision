import joblib
import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = 'model.joblib'

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        joblib.dump(model, MODEL_PATH)
    return model

def predict(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1]  # Return the name of the predicted class
