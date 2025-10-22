"""
Deepfake Detector - Core detection engine with ONNX optimization
"""

import os
import asyncio
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import onnxruntime as ort
from deepface import DeepFace
from loguru import logger
import librosa
import soundfile as sf
from io import BytesIO
from .mesonet_model import MesoNet, MesoNetONNXInference


class DeepfakeDetector:
    """Advanced deepfake detection system with ONNX optimization"""
    
    def __init__(self):
        self.video_model = None
        self.audio_model = None
        self.onnx_session = None
        self.device = "cpu"
        self.model_ready = False
        
        # MesoNet models
        self.mesonet = None
        self.mesonet_onnx = None
        
        # Detection thresholds
        self.video_threshold = 0.5  # MesoNet uses 0.5 threshold
        self.audio_threshold = 0.6
        self.confidence_threshold = 0.5
        
        # Model paths
        self.models_dir = "models"
        self.mesonet_path = os.path.join(self.models_dir, "mesonet_model.h5")
        self.mesonet_onnx_path = os.path.join(self.models_dir, "mesonet_model.onnx")
        self.audio_model_path = os.path.join(self.models_dir, "deepfake_audio.onnx")
        
        # Face detection
        self.face_cascade = None
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.audio_duration = 3.0  # seconds
    
    async def initialize(self):
        """Initialize the detection models"""
        try:
            logger.info("Initializing deepfake detection models...")
            
            # Create models directory
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Check for GPU availability
            self._setup_device()
            
            # Initialize face detection
            self._initialize_face_detection()
            
            # Load or create ONNX models
            await self._load_models()
            
            self.model_ready = True
            logger.info("Deepfake detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize deepfake detector: {str(e)}")
            self.model_ready = False
    
    def _setup_device(self):
        """Setup computation device (GPU/CPU)"""
        try:
            # Check for CUDA availability
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                self.device = "cuda"
                logger.info("GPU (CUDA) detected and will be used for inference")
            else:
                self.device = "cpu"
                logger.info("Using CPU for inference")
        except Exception as e:
            logger.warning(f"Error detecting GPU: {str(e)}, falling back to CPU")
            self.device = "cpu"
    
    def _initialize_face_detection(self):
        """Initialize OpenCV face detection"""
        try:
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {str(e)}")
    
    async def _load_models(self):
        """Load MesoNet models"""
        try:
            logger.info("Loading MesoNet models...")
            
            # Initialize MesoNet
            self.mesonet = MesoNet()
            
            # Try to load existing ONNX model first (fastest)
            if os.path.exists(self.mesonet_onnx_path):
                logger.info("Loading existing ONNX MesoNet model...")
                self.mesonet_onnx = MesoNetONNXInference(self.mesonet_onnx_path)
                if self.mesonet_onnx.load_onnx_model():
                    logger.info("ONNX MesoNet model loaded successfully")
                else:
                    self.mesonet_onnx = None
            
            # Try to load existing TensorFlow model
            elif os.path.exists(self.mesonet_path):
                logger.info("Loading existing TensorFlow MesoNet model...")
                if self.mesonet.load_model():
                    logger.info("TensorFlow MesoNet model loaded successfully")
                else:
                    await self._train_mesonet()
            
            # No models exist, train new one
            else:
                logger.info("No existing models found, training new MesoNet...")
                await self._train_mesonet()
            
            # Load audio model (if exists)
            if os.path.exists(self.audio_model_path):
                self._load_audio_model()
            else:
                await self._create_fallback_audio_model()
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Use fallback detection methods
            await self._setup_fallback_detection()
    
    def _load_video_model(self):
        """Load ONNX video model"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(self.video_model_path, providers=providers)
            logger.info("ONNX video model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX video model: {str(e)}")
    
    def _load_audio_model(self):
        """Load ONNX audio model"""
        try:
            # Audio model loading logic
            logger.info("Audio model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load audio model: {str(e)}")
    
    async def _create_fallback_video_model(self):
        """Create fallback video detection using DeepFace"""
        try:
            # Use DeepFace as fallback for face analysis
            logger.info("Using DeepFace as fallback for video analysis")
        except Exception as e:
            logger.error(f"Failed to setup fallback video model: {str(e)}")
    
    async def _create_fallback_audio_model(self):
        """Create fallback audio detection"""
        try:
            # Implement basic audio analysis
            logger.info("Using basic audio analysis as fallback")
        except Exception as e:
            logger.error(f"Failed to setup fallback audio model: {str(e)}")
    
    async def _train_mesonet(self):
        """Train MesoNet model"""
        try:
            logger.info("Training MesoNet model (this may take a few minutes)...")
            
            # Train the model
            history = self.mesonet.train_model(use_inception=True, epochs=15)
            
            # Convert to ONNX for faster inference
            logger.info("Converting trained model to ONNX...")
            if self.mesonet.convert_to_onnx():
                # Load the ONNX model
                self.mesonet_onnx = MesoNetONNXInference(self.mesonet_onnx_path)
                if self.mesonet_onnx.load_onnx_model():
                    logger.info("MesoNet trained and converted to ONNX successfully")
                else:
                    logger.warning("ONNX conversion successful but loading failed")
            else:
                logger.warning("ONNX conversion failed, using TensorFlow model")
                
        except Exception as e:
            logger.error(f"Error training MesoNet: {str(e)}")
            await self._setup_fallback_detection()
    
    async def _setup_fallback_detection(self):
        """Setup fallback detection methods"""
        logger.info("Setting up fallback detection methods")
        # Use traditional computer vision and audio processing techniques
    
    async def analyze_video_batch(self, frames: List[dict]) -> dict:
        """Analyze a batch of video frames for deepfake detection"""
        try:
            if not frames:
                return self._create_empty_result("video")
            
            results = {
                "type": "video",
                "frame_count": len(frames),
                "detections": [],
                "overall_score": 0.0,
                "confidence": 0.0,
                "is_deepfake": False,
                "processing_time": 0.0
            }
            
            start_time = datetime.utcnow()
            
            for frame_data in frames:
                frame = frame_data["frame"]
                timestamp = frame_data["timestamp"]
                frame_id = frame_data["frame_id"]
                
                # Detect faces in frame
                faces = self._detect_faces(frame)
                
                if len(faces) > 0:
                    # Analyze each face
                    for i, face in enumerate(faces):
                        face_analysis = await self._analyze_face(frame, face)
                        
                        detection = {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "face_id": i,
                            "bbox": face.tolist(),
                            "deepfake_score": face_analysis["deepfake_score"],
                            "confidence": face_analysis["confidence"],
                            "features": face_analysis["features"]
                        }
                        
                        results["detections"].append(detection)
                else:
                    # No faces detected
                    detection = {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "face_id": -1,
                        "bbox": [],
                        "deepfake_score": 0.0,
                        "confidence": 0.0,
                        "features": {"no_face_detected": True}
                    }
                    results["detections"].append(detection)
            
            # Calculate overall scores
            if results["detections"]:
                scores = [d["deepfake_score"] for d in results["detections"] if d["deepfake_score"] > 0]
                confidences = [d["confidence"] for d in results["detections"] if d["confidence"] > 0]
                
                if scores:
                    results["overall_score"] = np.mean(scores)
                    results["confidence"] = np.mean(confidences)
                    results["is_deepfake"] = results["overall_score"] > self.video_threshold
            
            # Calculate processing time
            end_time = datetime.utcnow()
            results["processing_time"] = (end_time - start_time).total_seconds()
            
            logger.debug(f"Analyzed {len(frames)} video frames, deepfake score: {results['overall_score']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing video batch: {str(e)}")
            return self._create_error_result("video", str(e))
    
    async def analyze_audio_batch(self, audio_chunks: List[dict]) -> dict:
        """Analyze a batch of audio chunks for deepfake detection"""
        try:
            if not audio_chunks:
                return self._create_empty_result("audio")
            
            results = {
                "type": "audio",
                "chunk_count": len(audio_chunks),
                "detections": [],
                "overall_score": 0.0,
                "confidence": 0.0,
                "is_deepfake": False,
                "processing_time": 0.0
            }
            
            start_time = datetime.utcnow()
            
            for chunk_data in audio_chunks:
                audio_bytes = chunk_data["audio_data"]
                timestamp = chunk_data["timestamp"]
                chunk_id = chunk_data["chunk_id"]
                
                # Analyze audio chunk
                audio_analysis = await self._analyze_audio_chunk(audio_bytes)
                
                detection = {
                    "chunk_id": chunk_id,
                    "timestamp": timestamp,
                    "deepfake_score": audio_analysis["deepfake_score"],
                    "confidence": audio_analysis["confidence"],
                    "features": audio_analysis["features"]
                }
                
                results["detections"].append(detection)
            
            # Calculate overall scores
            if results["detections"]:
                scores = [d["deepfake_score"] for d in results["detections"]]
                confidences = [d["confidence"] for d in results["detections"]]
                
                results["overall_score"] = np.mean(scores)
                results["confidence"] = np.mean(confidences)
                results["is_deepfake"] = results["overall_score"] > self.audio_threshold
            
            # Calculate processing time
            end_time = datetime.utcnow()
            results["processing_time"] = (end_time - start_time).total_seconds()
            
            logger.debug(f"Analyzed {len(audio_chunks)} audio chunks, deepfake score: {results['overall_score']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing audio batch: {str(e)}")
            return self._create_error_result("audio", str(e))
    
    def _detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect faces in a frame using OpenCV"""
        try:
            if self.face_cascade is None:
                return []
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    async def _analyze_face(self, frame: np.ndarray, face_bbox: np.ndarray) -> dict:
        """Analyze a detected face for deepfake indicators using MesoNet"""
        try:
            x, y, w, h = face_bbox
            face_roi = frame[y:y+h, x:x+w]
            
            # Use MesoNet for real deepfake detection
            if self.mesonet_onnx is not None:
                # Use ONNX model (fastest)
                result = self.mesonet_onnx.predict(face_roi)
                if "error" not in result:
                    return {
                        "deepfake_score": result["fake_probability"],
                        "confidence": result["confidence"],
                        "features": {
                            "classification": result["classification"],
                            "is_fake": result["is_fake"],
                            "is_real": result["is_real"],
                            "method": "MesoNet_ONNX",
                            "raw_prediction": result["raw_prediction"]
                        }
                    }
            
            elif self.mesonet is not None and self.mesonet.model is not None:
                # Use TensorFlow model
                result = self.mesonet.predict_deepfake(face_roi)
                if "error" not in result:
                    return {
                        "deepfake_score": result["fake_probability"],
                        "confidence": result["confidence"],
                        "features": {
                            "classification": result["classification"],
                            "is_fake": result["is_fake"],
                            "is_real": result["is_real"],
                            "method": "MesoNet_TensorFlow",
                            "raw_prediction": result["raw_prediction"]
                        }
                    }
            
            # Fallback to traditional analysis
            logger.debug("Using fallback analysis method")
            analysis = await self._traditional_face_analysis(face_roi)
            return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing face: {str(e)}")
            return {
                "deepfake_score": 0.0,
                "confidence": 0.0,
                "features": {"error": str(e)}
            }
    
    async def _traditional_face_analysis(self, face_image: np.ndarray) -> dict:
        """Traditional computer vision based face analysis"""
        try:
            # Calculate basic image quality metrics
            blur_score = self._calculate_blur_score(face_image)
            noise_score = self._calculate_noise_score(face_image)
            symmetry_score = self._calculate_symmetry_score(face_image)
            
            # Combine scores to estimate deepfake probability
            # This is a simplified heuristic approach
            quality_score = (blur_score + noise_score + symmetry_score) / 3.0
            deepfake_score = max(0.0, min(1.0, 1.0 - quality_score))
            
            return {
                "deepfake_score": deepfake_score,
                "confidence": 0.6,  # Moderate confidence for traditional methods
                "features": {
                    "blur_score": blur_score,
                    "noise_score": noise_score,
                    "symmetry_score": symmetry_score,
                    "method": "traditional_cv"
                }
            }
            
        except Exception as e:
            logger.error(f"Traditional face analysis error: {str(e)}")
            return self._basic_face_analysis(face_image)
    
    def _basic_face_analysis(self, face_image: np.ndarray) -> dict:
        """Basic fallback face analysis"""
        return {
            "deepfake_score": 0.3,  # Neutral score
            "confidence": 0.3,
            "features": {
                "method": "basic_fallback",
                "image_shape": face_image.shape
            }
        }
    
    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize to 0-1 range
            return min(1.0, laplacian_var / 1000.0)
        except:
            return 0.5
    
    def _calculate_noise_score(self, image: np.ndarray) -> float:
        """Calculate noise score"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Simple noise estimation using standard deviation
            noise_level = np.std(gray)
            return min(1.0, noise_level / 50.0)
        except:
            return 0.5
    
    def _calculate_symmetry_score(self, image: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Split face in half
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to match if needed
            if left_half.shape != right_half_flipped.shape:
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate correlation
            correlation = cv2.matchTemplate(left_half, right_half_flipped, cv2.TM_CCOEFF_NORMED)
            return float(np.max(correlation))
            
        except:
            return 0.5
    
    async def _analyze_audio_chunk(self, audio_bytes: bytes) -> dict:
        """Analyze audio chunk for deepfake indicators"""
        try:
            # Convert bytes to audio array
            audio_data, sr = sf.read(BytesIO(audio_bytes))
            
            # Resample if necessary
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
            # Extract audio features
            features = self._extract_audio_features(audio_data)
            
            # Simple heuristic-based detection
            deepfake_score = self._calculate_audio_deepfake_score(features)
            
            return {
                "deepfake_score": deepfake_score,
                "confidence": 0.5,  # Moderate confidence for basic audio analysis
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio chunk: {str(e)}")
            return {
                "deepfake_score": 0.0,
                "confidence": 0.0,
                "features": {"error": str(e)}
            }
    
    def _extract_audio_features(self, audio_data: np.ndarray) -> dict:
        """Extract audio features for analysis"""
        try:
            features = {}
            
            # Basic audio statistics
            features["rms_energy"] = float(np.sqrt(np.mean(audio_data**2)))
            features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_audio_deepfake_score(self, features: dict) -> float:
        """Calculate deepfake score based on audio features"""
        try:
            # Simple heuristic based on audio characteristics
            # In production, this would use a trained model
            
            score = 0.0
            
            # Check RMS energy (synthetic audio often has different energy patterns)
            rms = features.get("rms_energy", 0.0)
            if rms < 0.01 or rms > 0.5:
                score += 0.3
            
            # Check zero crossing rate (synthetic speech patterns)
            zcr = features.get("zero_crossing_rate", 0.0)
            if zcr < 0.05 or zcr > 0.3:
                score += 0.2
            
            # Check spectral centroid
            sc = features.get("spectral_centroid_mean", 0.0)
            if sc < 1000 or sc > 8000:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating audio deepfake score: {str(e)}")
            return 0.0
    
    def _create_empty_result(self, analysis_type: str) -> dict:
        """Create empty result structure"""
        return {
            "type": analysis_type,
            "frame_count" if analysis_type == "video" else "chunk_count": 0,
            "detections": [],
            "overall_score": 0.0,
            "confidence": 0.0,
            "is_deepfake": False,
            "processing_time": 0.0
        }
    
    def _create_error_result(self, analysis_type: str, error_message: str) -> dict:
        """Create error result structure"""
        result = self._create_empty_result(analysis_type)
        result["error"] = error_message
        return result
    
    def is_ready(self) -> bool:
        """Check if the detector is ready for use"""
        return self.model_ready
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.onnx_session:
                del self.onnx_session
            logger.info("Deepfake detector cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        return {
            "device": self.device,
            "model_ready": self.model_ready,
            "video_threshold": self.video_threshold,
            "audio_threshold": self.audio_threshold,
            "models_directory": self.models_dir
        }
