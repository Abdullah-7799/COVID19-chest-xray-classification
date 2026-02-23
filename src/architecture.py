import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Dropout, Input
)
from tensorflow.keras.models import Model

def check_gpu():
    """Checks and prints GPU availability."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU is available: {gpus}")
    else:
        print("⚠️ GPU not found. Running on CPU.")

def build_model(input_shape=(299, 299, 3), num_classes=4):
    """CNN Architecture based on the training history."""
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    
    # Dense Head
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model