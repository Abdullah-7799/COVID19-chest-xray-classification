import numpy as np
import tensorflow as tf
from PIL import Image
from model import create_dynamic_model

def create_empty_model():
    return create_dynamic_model(height=299, width=299, channels=1, classes=4)

def preprocess_image(image_path, height=299, width=299):
    img = Image.open(image_path).convert('L')
    img = img.resize((height, width))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array.astype(np.float32) / 255.0

def predict(image_path, model, labels):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_idx]
    return labels[predicted_idx], confidence

if __name__ == "__main__":
    print("WARNING: This repository contains model architecture only.")
    print("Pretrained weights are not included.")
    print("To use a trained model, you need to provide .h5 weights file separately.")