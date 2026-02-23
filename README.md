# COVID-19 Chest X-Ray Classification

This project implements a Convolutional Neural Network (CNN) to classify chest X-ray images into four categories: COVID-19, Normal, Pneumonia, and Viral Pneumonia.

## ✨ Features
- **High Accuracy:** Trained on the COVID-19 Radiography Dataset.
- **Modern UI:** Built with Streamlit for easy interaction.
- **Batch Processing:** Ability to upload and predict multiple images at once.
- **GPU Optimized:** Supports CUDA-enabled GPUs for fast inference.

## 🚀 Quick Start
1. Clone the repo:
   ```bash
   git clone [https://github.com/Abdullah-7799/COVID19-chest-xray-classification.git](https://github.com/Abdullah-7799/COVID19-chest-xray-classification.git)
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:

   ```bash
   streamlit run app.py
   ```

🏗️ Architecture
The model uses a custom CNN architecture with Global Average Pooling and Dropout layers to prevent overfitting.