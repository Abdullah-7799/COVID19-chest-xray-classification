import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def create_dynamic_model(height=299, width=299, channels=1, classes=4):
    inputs = tf.keras.Input(shape=(height, width, channels))
    
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.1)(x)
    
    # Block 2  
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    
    # Classifier
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model