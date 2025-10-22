"""
Auto-training script for MesoNet (no user input required)
"""

import os
import sys
from loguru import logger

# Add src to path
sys.path.append('src')

def main():
    """Auto training function"""
    logger.info("🚀 Auto-training MesoNet for REAL vs FAKE Person Detection")
    
    try:
        from src.mesonet_model import MesoNet
        
        # Initialize and train MesoNet
        logger.info("🧠 Initializing MesoNet...")
        mesonet = MesoNet()
        
        # Train model
        logger.info("📚 Training MesoNet model (this may take a few minutes)...")
        history = mesonet.train_model(use_inception=True, epochs=10)
        
        # Try ONNX conversion (optional)
        logger.info("🔄 Attempting ONNX conversion...")
        onnx_success = mesonet.convert_to_onnx()
        
        if onnx_success:
            logger.info("✅ ONNX conversion successful!")
        else:
            logger.info("⚠️ ONNX conversion skipped (tf2onnx not available)")
        
        # Test the model
        logger.info("🧪 Testing the trained model...")
        test_model(mesonet)
        
        logger.info("🎉 SUCCESS! MesoNet training completed!")
        logger.info("📁 Model files created:")
        logger.info("   - models/mesonet_model.h5 (TensorFlow)")
        if onnx_success:
            logger.info("   - models/mesonet_model.onnx (ONNX)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model(mesonet):
    """Test the trained model"""
    try:
        import numpy as np
        
        logger.info("Testing MesoNet predictions...")
        
        # Create test images
        test_images = []
        
        # Simulate real face (more structured)
        real_face = np.random.normal(0.5, 0.1, (256, 256, 3))
        real_face = np.clip(real_face, 0, 1) * 255
        test_images.append(("Real Face Simulation", real_face.astype(np.uint8)))
        
        # Simulate fake face (more random)
        fake_face = np.random.uniform(0, 1, (256, 256, 3)) * 255
        test_images.append(("Fake Face Simulation", fake_face.astype(np.uint8)))
        
        # Test predictions
        for name, image in test_images:
            result = mesonet.predict_deepfake(image)
            
            if "error" not in result:
                classification = result["classification"]
                confidence = result["confidence"]
                fake_prob = result["fake_probability"]
                
                logger.info(f"🔍 {name}:")
                logger.info(f"   Classification: {classification}")
                logger.info(f"   Confidence: {confidence:.3f}")
                logger.info(f"   Fake Probability: {fake_prob:.3f}")
            else:
                logger.error(f"❌ Prediction error for {name}: {result['error']}")
        
        logger.info("✅ Model testing completed!")
        
    except Exception as e:
        logger.error(f"❌ Error testing model: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting automatic MesoNet training...")
    success = main()
    
    if success:
        logger.info("🎉 Training completed successfully!")
        logger.info("🚀 You can now restart your TrueFace server to use the trained model")
    else:
        logger.error("❌ Training failed!")
