import tensorflow as tf
from .architecture import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def setup_data(data_path, img_size=(299, 299), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        fill_mode='nearest'
    )
    
    train_gen = datagen.flow_from_directory(
        data_path, target_size=img_size, batch_size=batch_size,
        subset='training', class_mode='categorical'
    )
    
    val_gen = datagen.flow_from_directory(
        data_path, target_size=img_size, batch_size=batch_size,
        subset='validation', class_mode='categorical'
    )
    return train_gen, val_gen

def train_new_model(data_path, epochs=20):
    train_gen, val_gen = setup_data(data_path)
    
    # Building model from scratch
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.weights.h5', save_best_only=True, save_weights_only=True
    )
    
    print("🚀 Starting fresh training...")
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint])