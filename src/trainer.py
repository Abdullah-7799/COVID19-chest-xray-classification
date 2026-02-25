import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from architecture import build_model

# Configuration
DATA_DIR = 'E:\Datasets\COVID-19_Radiography_Dataset' # Address of dataset
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 20

def train():
    # 1. Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        subset='training', class_mode='categorical'
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        subset='validation', class_mode='categorical'
    )

    # 2. Build and Compile Model
    model = build_model(input_shape=(299, 299, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.weights.h5', save_best_only=True, save_weights_only=True
    )

    # 4. Training
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])

if __name__ == "__main__":
    train()