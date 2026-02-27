import tensorflow as tf
from .architecture import build_model
from .trainer import setup_data

def resume_training(data_path, weights_path, initial_epoch=10, extra_epochs=10):
    """
    Loads existing weights and continues training.
    """
    train_gen, val_gen = setup_data(data_path)
    
    # 1. Reconstruct Architecture
    model = build_model()
    
    # 2. Load previous weights
    print(f"📦 Loading weights from {weights_path}...")
    model.load_weights(weights_path)
    
    # 3. Compile (Use a lower learning rate for fine-tuning)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/continued_model.weights.h5', save_best_only=True, save_weights_only=True
    )
    
    print(f"⏳ Resuming training from epoch {initial_epoch}...")
    model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=initial_epoch + extra_epochs, 
        initial_epoch=initial_epoch, 
        callbacks=[checkpoint]
    )

if __name__ == "__main__":
    # Example usage:
    # resume_training('data/', 'models/size_299_weights_val_acc0.900331.h5')
    pass