#!/usr/bin/env python3
"""
Advanced MesoNet Fine-Tuning Script
Optimized for your 190K+ image dataset with advanced techniques
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedMesoNetTrainer:
    def __init__(self, dataset_path="Dataset", model_save_path="models"):
        self.dataset_path = Path(dataset_path)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        # Training configuration
        self.config = {
            "image_size": (256, 256),
            "batch_size": 32,
            "initial_epochs": 20,
            "fine_tune_epochs": 30,
            "initial_learning_rate": 0.001,
            "fine_tune_learning_rate": 0.0001,
            "dropout_rate": 0.5,
            "l2_regularization": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
        }
        
        # Data augmentation for better generalization
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        self.val_datagen = ImageDataGenerator(rescale=1./255)
        
        logger.info("🚀 Advanced MesoNet Trainer initialized")
        logger.info(f"📁 Dataset path: {self.dataset_path}")
        logger.info(f"💾 Model save path: {self.model_save_path}")

    def create_improved_mesonet(self):
        """Create an improved MesoNet architecture with modern techniques"""
        
        inputs = keras.Input(shape=(*self.config["image_size"], 3))
        
        # Initial convolution with batch normalization
        x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # First MesoNet block
        x = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Second MesoNet block with increased filters
        x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Third block with more filters
        x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((4, 4), padding='same')(x)
        
        # Additional convolutional layers for better feature extraction
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers with dropout and L2 regularization
        x = layers.Dense(16, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(self.config["l2_regularization"]))(x)
        x = layers.Dropout(self.config["dropout_rate"])(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='prediction')(x)
        
        model = keras.Model(inputs, outputs, name='ImprovedMesoNet')
        
        logger.info("🧠 Improved MesoNet architecture created")
        logger.info(f"📊 Total parameters: {model.count_params():,}")
        
        return model

    def prepare_data_generators(self):
        """Prepare data generators for training"""
        
        train_generator = self.train_datagen.flow_from_directory(
            self.dataset_path / "Train",
            target_size=self.config["image_size"],
            batch_size=self.config["batch_size"],
            class_mode='binary',
            classes=['Real', 'Fake'],  # Real=0, Fake=1
            shuffle=True,
            seed=42
        )
        
        validation_generator = self.val_datagen.flow_from_directory(
            self.dataset_path / "Validation",
            target_size=self.config["image_size"],
            batch_size=self.config["batch_size"],
            class_mode='binary',
            classes=['Real', 'Fake'],
            shuffle=False,
            seed=42
        )
        
        test_generator = self.val_datagen.flow_from_directory(
            self.dataset_path / "Test",
            target_size=self.config["image_size"],
            batch_size=self.config["batch_size"],
            class_mode='binary',
            classes=['Real', 'Fake'],
            shuffle=False,
            seed=42
        )
        
        logger.info(f"📚 Training samples: {train_generator.samples}")
        logger.info(f"🔍 Validation samples: {validation_generator.samples}")
        logger.info(f"🧪 Test samples: {test_generator.samples}")
        
        return train_generator, validation_generator, test_generator

    def create_callbacks(self, stage="initial"):
        """Create training callbacks"""
        
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config["reduce_lr_patience"],
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=self.model_save_path / f"mesonet_best_{stage}.h5",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                filename=self.model_save_path / f"training_log_{stage}.csv",
                append=True
            )
        ]
        
        return callbacks_list

    def train_model(self):
        """Train the improved MesoNet model"""
        
        logger.info("🎯 Starting advanced MesoNet training...")
        
        # Prepare data
        train_gen, val_gen, test_gen = self.prepare_data_generators()
        
        # Create model
        model = self.create_improved_mesonet()
        
        # Compile model for initial training
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config["initial_learning_rate"]),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("📋 Model compiled for initial training")
        
        # Initial training
        logger.info("🚀 Phase 1: Initial training...")
        
        initial_callbacks = self.create_callbacks("initial")
        
        history_initial = model.fit(
            train_gen,
            epochs=self.config["initial_epochs"],
            validation_data=val_gen,
            callbacks=initial_callbacks,
            verbose=1
        )
        
        # Load best weights from initial training
        best_initial_model_path = self.model_save_path / "mesonet_best_initial.h5"
        if best_initial_model_path.exists():
            model.load_weights(str(best_initial_model_path))
            logger.info("✅ Loaded best weights from initial training")
        
        # Fine-tuning phase
        logger.info("🎯 Phase 2: Fine-tuning with lower learning rate...")
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config["fine_tune_learning_rate"]),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        fine_tune_callbacks = self.create_callbacks("fine_tune")
        
        history_fine_tune = model.fit(
            train_gen,
            epochs=self.config["fine_tune_epochs"],
            validation_data=val_gen,
            callbacks=fine_tune_callbacks,
            verbose=1
        )
        
        # Load best fine-tuned model
        best_fine_tune_model_path = self.model_save_path / "mesonet_best_fine_tune.h5"
        if best_fine_tune_model_path.exists():
            model.load_weights(str(best_fine_tune_model_path))
            logger.info("✅ Loaded best weights from fine-tuning")
        
        # Save final model
        final_model_path = self.model_save_path / "mesonet_model.h5"
        model.save(str(final_model_path))
        logger.info(f"💾 Final model saved: {final_model_path}")
        
        # Evaluate on test set
        logger.info("🧪 Evaluating on test set...")
        test_results = model.evaluate(test_gen, verbose=1)
        
        logger.info("📊 Test Results:")
        logger.info(f"   Test Loss: {test_results[0]:.4f}")
        logger.info(f"   Test Accuracy: {test_results[1]:.4f}")
        logger.info(f"   Test Precision: {test_results[2]:.4f}")
        logger.info(f"   Test Recall: {test_results[3]:.4f}")
        
        # Generate detailed predictions for analysis
        self.detailed_evaluation(model, test_gen)
        
        # Plot training history
        self.plot_training_history(history_initial, history_fine_tune)
        
        # Save training configuration
        self.save_training_config(test_results)
        
        return model, history_initial, history_fine_tune

    def detailed_evaluation(self, model, test_generator):
        """Perform detailed evaluation with confusion matrix and classification report"""
        
        logger.info("📈 Generating detailed evaluation...")
        
        # Reset generator
        test_generator.reset()
        
        # Get predictions
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Get true labels
        true_classes = test_generator.classes
        class_labels = ['Real', 'Fake']
        
        # Classification report
        report = classification_report(true_classes, predicted_classes, 
                                     target_names=class_labels, output_dict=True)
        
        logger.info("📋 Classification Report:")
        logger.info(f"   Real - Precision: {report['Real']['precision']:.4f}, Recall: {report['Real']['recall']:.4f}")
        logger.info(f"   Fake - Precision: {report['Fake']['precision']:.4f}, Recall: {report['Fake']['recall']:.4f}")
        logger.info(f"   Overall Accuracy: {report['accuracy']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix - MesoNet Fine-tuned')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.model_save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'test_accuracy': float(report['accuracy']),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.model_save_path / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("💾 Detailed evaluation saved")

    def plot_training_history(self, history_initial, history_fine_tune):
        """Plot training history"""
        
        # Combine histories
        epochs_initial = len(history_initial.history['loss'])
        epochs_fine_tune = len(history_fine_tune.history['loss'])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(range(1, epochs_initial + 1), history_initial.history['loss'], 
                       label='Initial Training', color='blue')
        axes[0, 0].plot(range(epochs_initial + 1, epochs_initial + epochs_fine_tune + 1), 
                       history_fine_tune.history['loss'], label='Fine-tuning', color='red')
        axes[0, 0].plot(range(1, epochs_initial + 1), history_initial.history['val_loss'], 
                       label='Initial Validation', color='lightblue', linestyle='--')
        axes[0, 0].plot(range(epochs_initial + 1, epochs_initial + epochs_fine_tune + 1), 
                       history_fine_tune.history['val_loss'], label='Fine-tune Validation', 
                       color='lightcoral', linestyle='--')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(range(1, epochs_initial + 1), history_initial.history['accuracy'], 
                       label='Initial Training', color='blue')
        axes[0, 1].plot(range(epochs_initial + 1, epochs_initial + epochs_fine_tune + 1), 
                       history_fine_tune.history['accuracy'], label='Fine-tuning', color='red')
        axes[0, 1].plot(range(1, epochs_initial + 1), history_initial.history['val_accuracy'], 
                       label='Initial Validation', color='lightblue', linestyle='--')
        axes[0, 1].plot(range(epochs_initial + 1, epochs_initial + epochs_fine_tune + 1), 
                       history_fine_tune.history['val_accuracy'], label='Fine-tune Validation', 
                       color='lightcoral', linestyle='--')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        axes[1, 0].plot(range(1, epochs_initial + 1), history_initial.history['precision'], 
                       label='Initial Training', color='blue')
        axes[1, 0].plot(range(epochs_initial + 1, epochs_initial + epochs_fine_tune + 1), 
                       history_fine_tune.history['precision'], label='Fine-tuning', color='red')
        axes[1, 0].plot(range(1, epochs_initial + 1), history_initial.history['val_precision'], 
                       label='Initial Validation', color='lightblue', linestyle='--')
        axes[1, 0].plot(range(epochs_initial + 1, epochs_initial + epochs_fine_tune + 1), 
                       history_fine_tune.history['val_precision'], label='Fine-tune Validation', 
                       color='lightcoral', linestyle='--')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall plot
        axes[1, 1].plot(range(1, epochs_initial + 1), history_initial.history['recall'], 
                       label='Initial Training', color='blue')
        axes[1, 1].plot(range(epochs_initial + 1, epochs_initial + epochs_fine_tune + 1), 
                       history_fine_tune.history['recall'], label='Fine-tuning', color='red')
        axes[1, 1].plot(range(1, epochs_initial + 1), history_initial.history['val_recall'], 
                       label='Initial Validation', color='lightblue', linestyle='--')
        axes[1, 1].plot(range(epochs_initial + 1, epochs_initial + epochs_fine_tune + 1), 
                       history_fine_tune.history['val_recall'], label='Fine-tune Validation', 
                       color='lightcoral', linestyle='--')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_save_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Training history plots saved")

    def save_training_config(self, test_results):
        """Save training configuration and results"""
        
        config_data = {
            'training_config': self.config,
            'test_results': {
                'test_loss': float(test_results[0]),
                'test_accuracy': float(test_results[1]),
                'test_precision': float(test_results[2]),
                'test_recall': float(test_results[3])
            },
            'model_info': {
                'architecture': 'ImprovedMesoNet',
                'training_date': datetime.now().isoformat(),
                'dataset_size': {
                    'train': 140002,
                    'validation': 39428,
                    'test': 10905
                }
            }
        }
        
        with open(self.model_save_path / 'training_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info("💾 Training configuration saved")

def main():
    """Main training function"""
    
    print("""
    🎯 Advanced MesoNet Fine-Tuning
    ===============================
    
    This script will fine-tune your MesoNet model using:
    - 140K training images
    - 39K validation images  
    - 10K test images
    
    Features:
    ✅ Advanced data augmentation
    ✅ Two-phase training (initial + fine-tuning)
    ✅ Early stopping and learning rate scheduling
    ✅ Comprehensive evaluation with metrics
    ✅ Training visualization and analysis
    
    """)
    
    try:
        # Initialize trainer
        trainer = AdvancedMesoNetTrainer()
        
        # Train model
        model, history_initial, history_fine_tune = trainer.train_model()
        
        print("""
        🎉 Fine-tuning completed successfully!
        
        📁 Generated files:
        - models/mesonet_model.h5 (Final optimized model)
        - models/training_history.png (Training plots)
        - models/confusion_matrix.png (Performance analysis)
        - models/evaluation_results.json (Detailed metrics)
        - models/training_config.json (Configuration)
        
        🚀 Your model is now ready for production use!
        """)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Fine-tuning completed! Your model should now perform significantly better.")
        print("🔄 Restart your TrueFace server to use the improved model.")
    else:
        print("\n❌ Fine-tuning failed. Check the error messages above.")
