#!/usr/bin/env python3
"""
Quick Fine-Tuning Script for MesoNet
Faster training for quick iterations and testing
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickMesoNetTrainer:
    def __init__(self, dataset_path="Dataset", model_save_path="models"):
        self.dataset_path = Path(dataset_path)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        # Quick training configuration
        self.config = {
            "image_size": (128, 128),  # Smaller for faster training
            "batch_size": 64,          # Larger batch for speed
            "epochs": 10,              # Fewer epochs for quick iteration
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "subset_size": 10000,      # Use subset for quick training
        }
        
        # Simple data augmentation
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=self.config["validation_split"]
        )
        
        logger.info("⚡ Quick MesoNet Trainer initialized")

    def create_lightweight_mesonet(self):
        """Create a lightweight MesoNet for quick training"""
        
        inputs = keras.Input(shape=(*self.config["image_size"], 3))
        
        # Lightweight architecture
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs, name='LightweightMesoNet')
        
        logger.info(f"🧠 Lightweight model created with {model.count_params():,} parameters")
        return model

    def prepare_quick_data(self):
        """Prepare data generators for quick training"""
        
        # Use subset for faster training
        train_generator = self.train_datagen.flow_from_directory(
            self.dataset_path / "Train",
            target_size=self.config["image_size"],
            batch_size=self.config["batch_size"],
            class_mode='binary',
            classes=['Real', 'Fake'],
            subset='training',
            shuffle=True,
            seed=42
        )
        
        validation_generator = self.train_datagen.flow_from_directory(
            self.dataset_path / "Train",
            target_size=self.config["image_size"],
            batch_size=self.config["batch_size"],
            class_mode='binary',
            classes=['Real', 'Fake'],
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        logger.info(f"📚 Quick training - Train: {train_generator.samples}, Val: {validation_generator.samples}")
        return train_generator, validation_generator

    def quick_train(self):
        """Quick training function"""
        
        logger.info("⚡ Starting quick MesoNet training...")
        
        # Prepare data
        train_gen, val_gen = self.prepare_quick_data()
        
        # Create model
        model = self.create_lightweight_mesonet()
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Simple callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            callbacks.ModelCheckpoint(
                filepath=str(self.model_save_path / "mesonet_quick.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            train_gen,
            epochs=self.config["epochs"],
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        model.save(str(self.model_save_path / "mesonet_quick_final.h5"))
        
        # Quick evaluation
        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        
        logger.info("📊 Quick Training Results:")
        logger.info(f"   Validation Loss: {val_loss:.4f}")
        logger.info(f"   Validation Accuracy: {val_accuracy:.4f}")
        
        return model, history

def main():
    """Main quick training function"""
    
    print("""
    ⚡ Quick MesoNet Fine-Tuning
    ===========================
    
    This is a fast training script for:
    - Quick model iterations
    - Testing new architectures
    - Rapid prototyping
    
    Features:
    ⚡ Faster training (smaller images, fewer epochs)
    📊 Basic performance metrics
    💾 Quick model checkpoints
    
    """)
    
    try:
        trainer = QuickMesoNetTrainer()
        model, history = trainer.quick_train()
        
        print("""
        ⚡ Quick training completed!
        
        📁 Files created:
        - models/mesonet_quick_final.h5
        
        🚀 Use this for quick testing, then run the advanced trainer for production.
        """)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Quick fine-tuning completed!")
    else:
        print("\n❌ Quick fine-tuning failed.")
