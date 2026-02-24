import numpy as np
from PIL import Image
from .architecture import build_model, check_gpu

class InferenceEngine:
    def __init__(self, weights_path):
        check_gpu()
        self.class_names = ['Normal', 'COVID-19', 'Pneumonia', 'Viral Pneumonia']
        self.model = build_model()
        self.model.load_weights(weights_path)

    def process_image(self, image_input):
        """Standardizes image for the model."""
        img = Image.open(image_input).convert('RGB')
        img = img.resize((299, 299))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_input):
        """Runs inference on a single image."""
        processed_img = self.process_image(image_input)
        preds = self.model.predict(processed_img, verbose=0)[0]
        class_idx = np.argmax(preds)
        return {
            "label": self.class_names[class_idx],
            "confidence": float(preds[class_idx]),
            "scores": dict(zip(self.class_names, preds.tolist()))
        }