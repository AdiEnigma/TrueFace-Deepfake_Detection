"""
Utility functions for TrueFace Backend
"""

import os
import json
import base64
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import numpy as np
import cv2
from loguru import logger


def setup_logging(config):
    """Setup logging configuration"""
    try:
        # Create logs directory
        log_dir = os.path.dirname(config.logging.log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        
        # Add console handler
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=config.logging.level,
            format=config.logging.format,
            colorize=True
        )
        
        # Add file handler
        logger.add(
            sink=config.logging.log_file,
            level=config.logging.level,
            format=config.logging.format,
            rotation=config.logging.rotation,
            retention=config.logging.retention,
            compression="zip"
        )
        
        logger.info("Logging configured successfully")
        
    except Exception as e:
        print(f"Failed to setup logging: {str(e)}")


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "logs",
        "models",
        "temp",
        "uploads"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")


def generate_session_id() -> str:
    """Generate a unique session ID"""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
    return f"session_{timestamp}_{random_part}"


def generate_stream_id() -> str:
    """Generate a unique stream ID"""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
    return f"stream_{timestamp}_{random_part}"


def encode_image_to_base64(image: np.ndarray, format: str = "jpg") -> str:
    """Encode OpenCV image to base64 string"""
    try:
        if format.lower() == "jpg":
            _, buffer = cv2.imencode('.jpg', image)
        elif format.lower() == "png":
            _, buffer = cv2.imencode('.png', image)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return ""


def decode_base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """Decode base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        return image
        
    except Exception as e:
        logger.error(f"Error decoding base64 to image: {str(e)}")
        return None


def resize_image(image: np.ndarray, target_size: tuple, maintain_aspect_ratio: bool = True) -> np.ndarray:
    """Resize image to target size"""
    try:
        if maintain_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create canvas and center the image
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, target_size)
            
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image


def calculate_image_hash(image: np.ndarray) -> str:
    """Calculate hash of an image for deduplication"""
    try:
        # Convert to grayscale and resize for consistent hashing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        
        # Calculate hash
        image_hash = hashlib.md5(resized.tobytes()).hexdigest()
        return image_hash
        
    except Exception as e:
        logger.error(f"Error calculating image hash: {str(e)}")
        return ""


def validate_json_structure(data: Dict, required_fields: List[str]) -> bool:
    """Validate JSON structure has required fields"""
    try:
        for field in required_fields:
            if field not in data:
                return False
        return True
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    try:
        # Remove or replace unsafe characters
        unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
        
    except Exception as e:
        logger.error(f"Error sanitizing filename: {str(e)}")
        return "unknown_file"


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    try:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"
    except Exception:
        return "0 B"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    try:
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
    except Exception:
        return "0s"


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format"""
    return datetime.utcnow().isoformat() + "Z"


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime object"""
    try:
        # Handle different timestamp formats
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        
        return datetime.fromisoformat(timestamp_str)
    except Exception as e:
        logger.error(f"Error parsing timestamp: {str(e)}")
        return None


def calculate_fps(frame_count: int, duration: float) -> float:
    """Calculate frames per second"""
    try:
        if duration > 0:
            return frame_count / duration
        return 0.0
    except Exception:
        return 0.0


def calculate_bitrate(data_size_bytes: int, duration: float) -> float:
    """Calculate bitrate in bits per second"""
    try:
        if duration > 0:
            return (data_size_bytes * 8) / duration
        return 0.0
    except Exception:
        return 0.0


async def async_retry(func, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Async retry decorator with exponential backoff"""
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = delay * (backoff ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
            await asyncio.sleep(wait_time)


def create_response_template(success: bool = True, message: str = "", data: Any = None) -> Dict:
    """Create standardized API response template"""
    return {
        "success": success,
        "message": message,
        "data": data,
        "timestamp": get_timestamp()
    }


def validate_websocket_message(message: Dict) -> tuple[bool, str]:
    """Validate WebSocket message format"""
    try:
        required_fields = ["type"]
        
        if not validate_json_structure(message, required_fields):
            return False, "Missing required field: type"
        
        message_type = message.get("type")
        
        # Validate based on message type
        if message_type == "video_frame":
            if "data" not in message:
                return False, "Missing video frame data"
        elif message_type == "audio_chunk":
            if "data" not in message:
                return False, "Missing audio chunk data"
        elif message_type == "start_stream":
            # Optional config validation
            pass
        elif message_type == "stop_stream":
            # No additional validation needed
            pass
        elif message_type == "ping":
            # No additional validation needed
            pass
        else:
            return False, f"Unknown message type: {message_type}"
        
        return True, "Valid message"
        
    except Exception as e:
        return False, f"Message validation error: {str(e)}"


def get_system_info() -> Dict:
    """Get system information"""
    try:
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {}


class PerformanceMonitor:
    """Simple performance monitoring utility"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.metrics[name] = {"start_time": datetime.utcnow()}
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        if name in self.metrics and "start_time" in self.metrics[name]:
            duration = (datetime.utcnow() - self.metrics[name]["start_time"]).total_seconds()
            self.metrics[name]["duration"] = duration
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict:
        """Get all collected metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
