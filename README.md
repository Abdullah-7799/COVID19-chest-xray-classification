# COVID-19 Chest X-ray Classification

**Validation Accuracy: 94.67%**

Lightweight CNN model for detecting COVID-19 and other lung abnormalities from chest X-rays.

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| COVID | 0.99 | 1.00 | 1.00 | 723 |
| Lung_Opacity | 0.96 | 0.93 | 0.95 | 1202 |
| Normal | 0.96 | 0.97 | 0.97 | 2038 |
| Viral Pneumonia | 0.95 | 1.00 | 0.97 | 269 |

| Metric | Value |
|--------|-------|
| Accuracy | 0.97 |
| Macro avg | 0.97 |
| Weighted avg | 0.97 |

## Classes
- COVID-19
- Lung Opacity  
- Normal
- Viral Pneumonia

## Model Architecture
- 3 convolutional blocks (32 → 64 → 128 filters)
- BatchNormalization & Dropout
- GlobalAveragePooling2D
- ~100k parameters

## Warning
**This repository contains model architecture only, not pre-trained weights.**

To use this model, you need to:
1. Train from scratch using your own data, OR
2. Contact me for access to pre-trained weights via separate agreement

## Quick Start (architecture only)

```
pip install -r requirements.txt
python inference.py
```

## Contact

For commercial use, pre-trained weights, or freelance projects:  
[LinkedIn](https://linkedin.com/in/abdullah-mahmoudian)