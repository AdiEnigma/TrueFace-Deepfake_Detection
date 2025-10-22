"""
Buffer Manager - Handles video/audio buffering and batch processing
"""

import asyncio
import base64
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2
from loguru import logger


class BufferManager:
    """Manages video and audio buffers for efficient processing"""
    
    def __init__(self):
        self.video_buffers: Dict[str, asyncio.Queue] = {}
        self.audio_buffers: Dict[str, asyncio.Queue] = {}
        self.buffer_metadata: Dict[str, dict] = {}
        
        # Configuration
        self.video_batch_size = 10  # Process every 10 frames
        self.audio_batch_size = 5   # Process every 5 audio chunks
        self.max_buffer_size = 100  # Maximum buffer size per session
        self.frame_skip_rate = 3    # Analyze every 3rd frame for efficiency
    
    def initialize_session_buffers(self, session_id: str):
        """Initialize buffers for a new session"""
        self.video_buffers[session_id] = asyncio.Queue(maxsize=self.max_buffer_size)
        self.audio_buffers[session_id] = asyncio.Queue(maxsize=self.max_buffer_size)
        self.buffer_metadata[session_id] = {
            "video_frame_count": 0,
            "audio_chunk_count": 0,
            "last_video_process": 0,
            "last_audio_process": 0,
            "created_at": datetime.utcnow()
        }
        logger.info(f"Initialized buffers for session: {session_id}")
    
    async def add_video_frame(self, session_id: str, frame_data: str, timestamp: str):
        """Add a video frame to the buffer"""
        if session_id not in self.video_buffers:
            self.initialize_session_buffers(session_id)
        
        try:
            # Decode base64 frame data
            frame_bytes = base64.b64decode(frame_data)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame_item = {
                    "frame": frame,
                    "timestamp": timestamp,
                    "frame_id": self.buffer_metadata[session_id]["video_frame_count"]
                }
                
                # Add to buffer (non-blocking)
                try:
                    self.video_buffers[session_id].put_nowait(frame_item)
                    self.buffer_metadata[session_id]["video_frame_count"] += 1
                except asyncio.QueueFull:
                    # Remove oldest frame if buffer is full
                    try:
                        self.video_buffers[session_id].get_nowait()
                        self.video_buffers[session_id].put_nowait(frame_item)
                        logger.warning(f"Video buffer full for {session_id}, dropped oldest frame")
                    except asyncio.QueueEmpty:
                        pass
            else:
                logger.error(f"Failed to decode video frame for session: {session_id}")
                
        except Exception as e:
            logger.error(f"Error adding video frame for {session_id}: {str(e)}")
    
    async def add_audio_chunk(self, session_id: str, audio_data: str, timestamp: str):
        """Add an audio chunk to the buffer"""
        if session_id not in self.audio_buffers:
            self.initialize_session_buffers(session_id)
        
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            audio_item = {
                "audio_data": audio_bytes,
                "timestamp": timestamp,
                "chunk_id": self.buffer_metadata[session_id]["audio_chunk_count"]
            }
            
            # Add to buffer (non-blocking)
            try:
                self.audio_buffers[session_id].put_nowait(audio_item)
                self.buffer_metadata[session_id]["audio_chunk_count"] += 1
            except asyncio.QueueFull:
                # Remove oldest chunk if buffer is full
                try:
                    self.audio_buffers[session_id].get_nowait()
                    self.audio_buffers[session_id].put_nowait(audio_item)
                    logger.warning(f"Audio buffer full for {session_id}, dropped oldest chunk")
                except asyncio.QueueEmpty:
                    pass
                    
        except Exception as e:
            logger.error(f"Error adding audio chunk for {session_id}: {str(e)}")
    
    def should_process_video(self, session_id: str) -> bool:
        """Check if video buffer is ready for processing"""
        if session_id not in self.buffer_metadata:
            return False
        
        metadata = self.buffer_metadata[session_id]
        frames_since_last = metadata["video_frame_count"] - metadata["last_video_process"]
        
        return frames_since_last >= self.video_batch_size
    
    def should_process_audio(self, session_id: str) -> bool:
        """Check if audio buffer is ready for processing"""
        if session_id not in self.buffer_metadata:
            return False
        
        metadata = self.buffer_metadata[session_id]
        chunks_since_last = metadata["audio_chunk_count"] - metadata["last_audio_process"]
        
        return chunks_since_last >= self.audio_batch_size
    
    async def get_video_batch(self, session_id: str) -> List[dict]:
        """Get a batch of video frames for processing"""
        if session_id not in self.video_buffers:
            return []
        
        frames = []
        buffer = self.video_buffers[session_id]
        
        # Get frames with frame skipping for efficiency
        frame_count = 0
        collected_frames = 0
        
        while not buffer.empty() and collected_frames < self.video_batch_size:
            try:
                frame_item = buffer.get_nowait()
                
                # Apply frame skipping
                if frame_count % self.frame_skip_rate == 0:
                    frames.append(frame_item)
                    collected_frames += 1
                
                frame_count += 1
                
            except asyncio.QueueEmpty:
                break
        
        # Update metadata
        self.buffer_metadata[session_id]["last_video_process"] = \
            self.buffer_metadata[session_id]["video_frame_count"]
        
        logger.debug(f"Retrieved {len(frames)} video frames for processing from {session_id}")
        return frames
    
    async def get_audio_batch(self, session_id: str) -> List[dict]:
        """Get a batch of audio chunks for processing"""
        if session_id not in self.audio_buffers:
            return []
        
        chunks = []
        buffer = self.audio_buffers[session_id]
        
        # Get audio chunks
        while not buffer.empty() and len(chunks) < self.audio_batch_size:
            try:
                chunk_item = buffer.get_nowait()
                chunks.append(chunk_item)
            except asyncio.QueueEmpty:
                break
        
        # Update metadata
        self.buffer_metadata[session_id]["last_audio_process"] = \
            self.buffer_metadata[session_id]["audio_chunk_count"]
        
        logger.debug(f"Retrieved {len(chunks)} audio chunks for processing from {session_id}")
        return chunks
    
    async def flush_session_buffers(self, session_id: str):
        """Process remaining items in session buffers"""
        if session_id not in self.video_buffers:
            return
        
        # Process remaining video frames
        remaining_frames = []
        while not self.video_buffers[session_id].empty():
            try:
                frame_item = self.video_buffers[session_id].get_nowait()
                remaining_frames.append(frame_item)
            except asyncio.QueueEmpty:
                break
        
        # Process remaining audio chunks
        remaining_audio = []
        while not self.audio_buffers[session_id].empty():
            try:
                chunk_item = self.audio_buffers[session_id].get_nowait()
                remaining_audio.append(chunk_item)
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Flushed {len(remaining_frames)} video frames and {len(remaining_audio)} audio chunks for {session_id}")
        
        return {
            "remaining_video_frames": len(remaining_frames),
            "remaining_audio_chunks": len(remaining_audio)
        }
    
    def cleanup_session_buffers(self, session_id: str):
        """Clean up buffers for a session"""
        if session_id in self.video_buffers:
            del self.video_buffers[session_id]
        if session_id in self.audio_buffers:
            del self.audio_buffers[session_id]
        if session_id in self.buffer_metadata:
            del self.buffer_metadata[session_id]
        
        logger.info(f"Cleaned up buffers for session: {session_id}")
    
    def get_buffer_stats(self, session_id: str) -> dict:
        """Get buffer statistics for a session"""
        if session_id not in self.buffer_metadata:
            return {}
        
        video_size = self.video_buffers[session_id].qsize() if session_id in self.video_buffers else 0
        audio_size = self.audio_buffers[session_id].qsize() if session_id in self.audio_buffers else 0
        
        return {
            "video_buffer_size": video_size,
            "audio_buffer_size": audio_size,
            "total_video_frames": self.buffer_metadata[session_id]["video_frame_count"],
            "total_audio_chunks": self.buffer_metadata[session_id]["audio_chunk_count"],
            "last_video_process": self.buffer_metadata[session_id]["last_video_process"],
            "last_audio_process": self.buffer_metadata[session_id]["last_audio_process"]
        }
