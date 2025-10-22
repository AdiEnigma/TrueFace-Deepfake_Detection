"""
Test client for TrueFace WebSocket API
"""

import asyncio
import json
import base64
import cv2
import numpy as np
import websockets
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrueFaceTestClient:
    """Test client for TrueFace WebSocket API"""
    
    def __init__(self, server_url: str = "ws://localhost:8000/ws"):
        self.server_url = server_url
        self.websocket = None
        self.session_id = None
        self.running = False
    
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            logger.info(f"Connected to {self.server_url}")
            
            # Wait for connection confirmation
            response = await self.websocket.recv()
            message = json.loads(response)
            
            if message.get("type") == "connection_established":
                self.session_id = message.get("session_id")
                logger.info(f"Session established: {self.session_id}")
                return True
            else:
                logger.error(f"Unexpected response: {message}")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from server")
    
    async def send_message(self, message: dict):
        """Send a message to the server"""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
    
    async def start_stream(self, config: dict = None):
        """Start streaming session"""
        if not config:
            config = {
                "video_enabled": True,
                "audio_enabled": False,
                "quality": "medium",
                "detection_sensitivity": "medium"
            }
        
        message = {
            "type": "start_stream",
            "config": config
        }
        
        await self.send_message(message)
        logger.info("Stream started")
    
    async def stop_stream(self):
        """Stop streaming session"""
        message = {"type": "stop_stream"}
        await self.send_message(message)
        logger.info("Stream stopped")
    
    async def send_test_frame(self):
        """Send a test video frame"""
        # Create a test image (640x480, blue background with white text)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (255, 100, 100)  # Blue background
        
        # Add timestamp text
        timestamp_text = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(frame, f"Test Frame {timestamp_text}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame
        message = {
            "type": "video_frame",
            "data": frame_base64,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_message(message)
    
    async def send_ping(self):
        """Send ping message"""
        message = {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_message(message)
    
    async def listen_for_responses(self):
        """Listen for server responses"""
        try:
            while self.running and self.websocket:
                response = await self.websocket.recv()
                message = json.loads(response)
                
                message_type = message.get("type")
                
                if message_type == "video_analysis_result":
                    results = message.get("results", {})
                    logger.info(f"Video Analysis - Score: {results.get('overall_score', 0):.3f}, "
                              f"Deepfake: {results.get('is_deepfake', False)}, "
                              f"Confidence: {results.get('confidence', 0):.3f}")
                
                elif message_type == "audio_analysis_result":
                    results = message.get("results", {})
                    logger.info(f"Audio Analysis - Score: {results.get('overall_score', 0):.3f}, "
                              f"Deepfake: {results.get('is_deepfake', False)}")
                
                elif message_type == "stream_started":
                    logger.info("Stream started confirmation received")
                
                elif message_type == "stream_stopped":
                    logger.info("Stream stopped confirmation received")
                    summary = message.get("summary", {})
                    logger.info(f"Session summary: {summary}")
                
                elif message_type == "pong":
                    logger.info("Pong received")
                
                elif message_type == "error":
                    logger.error(f"Server error: {message.get('message')}")
                
                else:
                    logger.info(f"Received: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")
        except Exception as e:
            logger.error(f"Error listening for responses: {str(e)}")
    
    async def run_test_session(self, duration: int = 30, frame_interval: float = 1.0):
        """Run a test session"""
        try:
            # Connect to server
            if not await self.connect():
                return
            
            self.running = True
            
            # Start listening for responses
            listen_task = asyncio.create_task(self.listen_for_responses())
            
            # Start stream
            await self.start_stream()
            await asyncio.sleep(1)
            
            # Send test frames
            logger.info(f"Sending test frames for {duration} seconds...")
            start_time = asyncio.get_event_loop().time()
            
            while (asyncio.get_event_loop().time() - start_time) < duration:
                await self.send_test_frame()
                await asyncio.sleep(frame_interval)
                
                # Send ping every 10 frames
                if int((asyncio.get_event_loop().time() - start_time) / frame_interval) % 10 == 0:
                    await self.send_ping()
            
            # Stop stream
            await self.stop_stream()
            await asyncio.sleep(2)
            
            # Stop listening
            self.running = False
            listen_task.cancel()
            
            # Disconnect
            await self.disconnect()
            
            logger.info("Test session completed")
            
        except Exception as e:
            logger.error(f"Test session error: {str(e)}")
        finally:
            self.running = False


async def main():
    """Main test function"""
    logger.info("Starting TrueFace WebSocket test client...")
    
    client = TrueFaceTestClient()
    
    # Run test session
    await client.run_test_session(duration=30, frame_interval=2.0)


if __name__ == "__main__":
    asyncio.run(main())
