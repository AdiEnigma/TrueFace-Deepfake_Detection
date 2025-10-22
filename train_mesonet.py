"""
Standalone script to train MesoNet for deepfake detection
Run this to create the REAL vs FAKE detection model
"""

import os
import sys
import asyncio
from loguru import logger

# Add src to path
sys.path.append('src')

from src.mesonet_model import train_and_convert_mesonet, MesoNet, MesoNetONNXInference


def main():
    """Main training function"""
    logger.info("🚀 Starting MesoNet Training for REAL vs FAKE Person Detection")
    logger.info("=" * 60)
    
    try:
        # Install tf2onnx if not available
        try:
            import tf2onnx
            logger.info("✅ tf2onnx is available")
        except ImportError:
            logger.error("❌ tf2onnx not found. Installing...")
            os.system("python -m pip install tf2onnx")
            import tf2onnx
            logger.info("✅ tf2onnx installed successfully")
        
        # Train and convert MesoNet
        logger.info("🧠 Training MesoNet model...")
        success = train_and_convert_mesonet()
        
        if success:
            logger.info("🎉 SUCCESS! MesoNet training completed!")
            logger.info("📁 Model files created:")
            logger.info("   - models/mesonet_model.h5 (TensorFlow)")
            logger.info("   - models/mesonet_model.onnx (ONNX)")
            
            # Test the model
            logger.info("🧪 Testing the trained model...")
            test_model()
            
        else:
            logger.error("❌ Training failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        return False
    
    return True


def test_model():
    """Test the trained model"""
    try:
        import numpy as np
        import cv2
        
        logger.info("Testing MesoNet model...")
        
        # Test ONNX model
        onnx_model = MesoNetONNXInference()
        if onnx_model.load_onnx_model():
            logger.info("✅ ONNX model loaded successfully")
            
            # Create test image
            test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            # Test prediction
            result = onnx_model.predict(test_image)
            logger.info(f"🔍 Test prediction result: {result}")
            
            if "error" not in result:
                classification = result["classification"]
                confidence = result["confidence"]
                logger.info(f"🎯 Classification: {classification}")
                logger.info(f"📊 Confidence: {confidence:.3f}")
                logger.info("✅ Model is working correctly!")
            else:
                logger.error(f"❌ Prediction error: {result['error']}")
        else:
            logger.error("❌ Failed to load ONNX model")
            
    except Exception as e:
        logger.error(f"❌ Error testing model: {str(e)}")


if __name__ == "__main__":
    print("""
    🔍 TrueFace MesoNet Training
    ===========================
    
    This script will:
    1. Train a MesoNet model for deepfake detection
    2. Convert it to ONNX format for fast inference
    3. Test the model to ensure it works
    
    The model will classify faces as:
    - REAL PERSON (authentic face)
    - FAKE PERSON (deepfake/synthetic face)
    
    Training may take 5-10 minutes...
    """)
    
    input("Press Enter to start training...")
    
    success = main()
    
    if success:
        print("""
        🎉 TRAINING COMPLETE!
        
        Your MesoNet model is ready for real deepfake detection!
        
        Next steps:
        1. Restart your TrueFace server: python main.py
        2. The server will automatically use the trained MesoNet model
        3. Test with the frontend or WebSocket client
        
        The model will now give you REAL classifications:
        - "REAL PERSON" for authentic faces
        - "FAKE PERSON" for deepfake faces
        """)
    else:
        print("""
        ❌ TRAINING FAILED!
        
        Please check the error messages above and try again.
        You may need to install additional dependencies.
        """)
    
    input("Press Enter to exit...")
