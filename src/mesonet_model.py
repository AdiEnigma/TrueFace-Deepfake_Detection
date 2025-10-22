"""
MesoNet Implementation for Deepfake Detection
Real vs Fake Person Detection using MesoNet architecture
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from loguru import logger
from typing import Tuple, Optional
import requests
import zipfile


class MesoNet:
    """MesoNet architecture for deepfake detection"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3)):
        self.input_shape = input_shape
        self.model = None
        self.model_path = "models/mesonet_model.h5"
        self.onnx_path = "models/mesonet_model.onnx"
        
    def build_meso4_model(self) -> keras.Model:
        """Build MesoNet-4 architecture"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            
            # Second convolutional block
            layers.Conv2D(8, (5, 5), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            
            # Third convolutional block
            layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            
            # Fourth convolutional block
            layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification: Real (0) or Fake (1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_mesonet_inception_model(self) -> keras.Model:
        """Build MesoNet with Inception modules"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Inception-like module
        def inception_module(x, filters):
            # 1x1 conv
            conv1x1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
            
            # 3x3 conv
            conv3x3 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            
            # 5x5 conv (replaced with two 3x3 for efficiency)
            conv5x5 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            conv5x5 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(conv5x5)
            
            # Max pooling branch
            pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
            pool = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(pool)
            
            # Concatenate all branches
            output = layers.concatenate([conv1x1, conv3x3, conv5x5, pool], axis=-1)
            return output
        
        # Initial convolution
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        # Inception modules
        x = inception_module(x, 8)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = inception_module(x, 16)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = inception_module(x, 32)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        # Final layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_synthetic_training_data(self, num_samples: int = 1000):
        """Create synthetic training data for demonstration"""
        logger.info(f"Creating synthetic training data with {num_samples} samples...")
        
        # Create synthetic real faces (more structured patterns)
        real_faces = []
        fake_faces = []
        
        for i in range(num_samples // 2):
            # Real face simulation (more natural patterns)
            real_face = np.random.normal(0.5, 0.1, self.input_shape)
            # Add some structure to simulate real face features
            real_face = cv2.GaussianBlur(real_face.astype(np.float32), (5, 5), 1.0)
            real_face = np.clip(real_face, 0, 1)
            real_faces.append(real_face)
            
            # Fake face simulation (more artificial patterns)
            fake_face = np.random.uniform(0, 1, self.input_shape)
            # Add artificial artifacts
            fake_face = cv2.medianBlur(fake_face.astype(np.float32), 3)
            # Add some noise to simulate compression artifacts
            noise = np.random.normal(0, 0.05, self.input_shape)
            fake_face = np.clip(fake_face + noise, 0, 1)
            fake_faces.append(fake_face)
        
        # Combine data
        X = np.array(real_faces + fake_faces)
        y = np.array([0] * len(real_faces) + [1] * len(fake_faces))  # 0 = Real, 1 = Fake
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def train_model(self, use_inception: bool = False, epochs: int = 10):
        """Train the MesoNet model"""
        logger.info("Training MesoNet model...")
        
        # Build model
        if use_inception:
            self.model = self.build_mesonet_inception_model()
            logger.info("Using MesoNet with Inception modules")
        else:
            self.model = self.build_meso4_model()
            logger.info("Using MesoNet-4 architecture")
        
        # Create training data
        X_train, y_train = self.create_synthetic_training_data(2000)
        X_val, y_val = self.create_synthetic_training_data(400)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return history
    
    def load_model(self) -> bool:
        """Load pre-trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def convert_to_onnx(self):
        """Convert TensorFlow model to ONNX format"""
        try:
            if self.model is None:
                logger.error("No model loaded. Train or load a model first.")
                return False
            
            logger.info("Converting model to ONNX format...")
            
            try:
                # Try to import tf2onnx
                import tf2onnx
                
                # Create a sample input for conversion
                sample_input = np.random.random((1, *self.input_shape)).astype(np.float32)
                
                # Get the concrete function
                concrete_func = tf.function(lambda x: self.model(x))
                concrete_func = concrete_func.get_concrete_function(
                    tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32)
                )
                
                # Convert to ONNX
                onnx_model, _ = tf2onnx.convert.from_function(
                    concrete_func,
                    input_signature=[tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32)],
                    opset=13
                )
                
                # Save ONNX model
                os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)
                with open(self.onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
                logger.info(f"ONNX model saved to {self.onnx_path}")
                return True
                
            except ImportError:
                logger.warning("tf2onnx not available, skipping ONNX conversion")
                logger.info("You can install tf2onnx later with: pip install tf2onnx")
                return False
            
        except Exception as e:
            logger.error(f"Error converting to ONNX: {str(e)}")
            return False
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for prediction"""
        # Resize to model input size
        face_resized = cv2.resize(face_image, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize to [0, 1]
        if face_resized.dtype == np.uint8:
            face_normalized = face_resized.astype(np.float32) / 255.0
        else:
            face_normalized = face_resized.astype(np.float32)
        
        # Ensure 3 channels
        if len(face_normalized.shape) == 2:
            face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_GRAY2RGB)
        elif face_normalized.shape[2] == 4:
            face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_BGRA2RGB)
        
        # Add batch dimension
        return np.expand_dims(face_normalized, axis=0)
    
    def predict_deepfake(self, face_image: np.ndarray) -> dict:
        """Predict if face is real or fake"""
        try:
            if self.model is None:
                logger.error("No model loaded")
                return {"error": "No model loaded"}
            
            # Preprocess image
            processed_face = self.preprocess_face(face_image)
            
            # Make prediction
            prediction = self.model.predict(processed_face, verbose=0)[0][0]
            
            # Convert to probability and classification
            fake_probability = float(prediction)
            real_probability = 1.0 - fake_probability
            
            # Determine classification (threshold at 0.5)
            is_fake = fake_probability > 0.5
            confidence = max(fake_probability, real_probability)
            
            result = {
                "is_fake": is_fake,
                "is_real": not is_fake,
                "fake_probability": fake_probability,
                "real_probability": real_probability,
                "confidence": confidence,
                "classification": "FAKE PERSON" if is_fake else "REAL PERSON",
                "raw_prediction": prediction
            }
            
            logger.debug(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {"error": str(e)}
    
    def get_model_info(self) -> dict:
        """Get model information"""
        info = {
            "model_loaded": self.model is not None,
            "input_shape": self.input_shape,
            "model_path": self.model_path,
            "onnx_path": self.onnx_path,
            "model_exists": os.path.exists(self.model_path),
            "onnx_exists": os.path.exists(self.onnx_path)
        }
        
        if self.model is not None:
            info["model_summary"] = {
                "total_params": self.model.count_params(),
                "trainable_params": sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                "layers": len(self.model.layers)
            }
        
        return info


class MesoNetONNXInference:
    """ONNX Runtime inference for MesoNet"""
    
    def __init__(self, onnx_path: str = "models/mesonet_model.onnx"):
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.output_name = None
    
    def load_onnx_model(self) -> bool:
        """Load ONNX model for inference"""
        try:
            import onnxruntime as ort
            
            if not os.path.exists(self.onnx_path):
                logger.error(f"ONNX model not found: {self.onnx_path}")
                return False
            
            # Create inference session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"ONNX model loaded successfully: {self.onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            return False
    
    def predict(self, face_image: np.ndarray) -> dict:
        """Predict using ONNX model"""
        try:
            if self.session is None:
                return {"error": "ONNX model not loaded"}
            
            # Preprocess (same as TensorFlow version)
            mesonet = MesoNet()
            processed_face = mesonet.preprocess_face(face_image)
            
            # Run inference
            result = self.session.run([self.output_name], {self.input_name: processed_face})
            prediction = result[0][0][0]
            
            # Convert to result format
            fake_probability = float(prediction)
            real_probability = 1.0 - fake_probability
            is_fake = fake_probability > 0.5
            confidence = max(fake_probability, real_probability)
            
            return {
                "is_fake": is_fake,
                "is_real": not is_fake,
                "fake_probability": fake_probability,
                "real_probability": real_probability,
                "confidence": confidence,
                "classification": "FAKE PERSON" if is_fake else "REAL PERSON",
                "raw_prediction": prediction,
                "inference_engine": "ONNX Runtime"
            }
            
        except Exception as e:
            logger.error(f"ONNX inference error: {str(e)}")
            return {"error": str(e)}


# Training and conversion script
def train_and_convert_mesonet():
    """Train MesoNet and convert to ONNX"""
    logger.info("Starting MesoNet training and conversion process...")
    
    # Initialize MesoNet
    mesonet = MesoNet()
    
    # Train model
    history = mesonet.train_model(use_inception=True, epochs=20)
    
    # Convert to ONNX
    success = mesonet.convert_to_onnx()
    
    if success:
        logger.info("MesoNet training and ONNX conversion completed successfully!")
        
        # Test ONNX inference
        onnx_model = MesoNetONNXInference()
        if onnx_model.load_onnx_model():
            logger.info("ONNX model loaded and ready for inference")
        
        return True
    else:
        logger.error("ONNX conversion failed")
        return False


if __name__ == "__main__":
    # Run training and conversion
    train_and_convert_mesonet()
