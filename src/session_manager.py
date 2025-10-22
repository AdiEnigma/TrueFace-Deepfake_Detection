"""
Session Manager - Handles user sessions and stream management
"""

import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class SessionConfig:
    """Configuration for a session"""
    video_enabled: bool = True
    audio_enabled: bool = True
    frame_rate: int = 30
    audio_sample_rate: int = 16000
    quality: str = "medium"  # low, medium, high
    detection_sensitivity: str = "medium"  # low, medium, high
    real_time_alerts: bool = True


@dataclass
class SessionStats:
    """Statistics for a session"""
    total_frames_processed: int = 0
    total_audio_chunks_processed: int = 0
    total_detections: int = 0
    deepfake_detections: int = 0
    average_video_score: float = 0.0
    average_audio_score: float = 0.0
    session_duration: float = 0.0
    data_processed_mb: float = 0.0


@dataclass
class Session:
    """Session data structure"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    config: SessionConfig
    stats: SessionStats
    status: str = "active"  # active, paused, ended
    user_id: Optional[str] = None
    stream_id: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SessionManager:
    """Manages user sessions and their lifecycle"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Session] = {}
        self.session_history: List[str] = []
        self.cleanup_interval = 3600  # 1 hour
        self.max_session_duration = 86400  # 24 hours
        self.inactive_timeout = 1800  # 30 minutes
        
        # Initialize cleanup task as None (will be started when event loop is available)
        self._cleanup_task = None
    
    def _start_cleanup_task(self):
        """Start the session cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically cleanup inactive sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_inactive_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {str(e)}")
    
    async def _cleanup_inactive_sessions(self):
        """Remove inactive sessions"""
        current_time = datetime.utcnow()
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            # Check for inactivity
            time_since_activity = (current_time - session.last_activity).total_seconds()
            session_duration = (current_time - session.created_at).total_seconds()
            
            if (time_since_activity > self.inactive_timeout or 
                session_duration > self.max_session_duration):
                sessions_to_remove.append(session_id)
        
        # Remove inactive sessions
        for session_id in sessions_to_remove:
            await self.end_session(session_id, reason="timeout")
            logger.info(f"Cleaned up inactive session: {session_id}")
    
    async def create_session(self, session_id: str, user_id: Optional[str] = None, 
                           config: Optional[Dict] = None) -> Session:
        """Create a new session"""
        try:
            # Create session config
            session_config = SessionConfig()
            if config:
                for key, value in config.items():
                    if hasattr(session_config, key):
                        setattr(session_config, key, value)
            
            # Create session
            session = Session(
                session_id=session_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                config=session_config,
                stats=SessionStats(),
                user_id=user_id
            )
            
            # Store session
            self.active_sessions[session_id] = session
            self.session_history.append(session_id)
            
            logger.info(f"Created new session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {str(e)}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        session = self.active_sessions.get(session_id)
        if session:
            # Update last activity
            session.last_activity = datetime.utcnow()
        return session
    
    async def update_session_config(self, session_id: str, config: Dict) -> bool:
        """Update session configuration"""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return False
            
            # Update config
            for key, value in config.items():
                if hasattr(session.config, key):
                    setattr(session.config, key, value)
            
            logger.info(f"Updated config for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating session config {session_id}: {str(e)}")
            return False
    
    async def update_session_stats(self, session_id: str, stats_update: Dict) -> bool:
        """Update session statistics"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Update stats
            for key, value in stats_update.items():
                if hasattr(session.stats, key):
                    current_value = getattr(session.stats, key)
                    if isinstance(current_value, (int, float)):
                        # For numeric values, add to existing
                        if key.startswith('total_') or key.startswith('deepfake_'):
                            setattr(session.stats, key, current_value + value)
                        else:
                            # For averages, update with new value
                            setattr(session.stats, key, value)
                    else:
                        setattr(session.stats, key, value)
            
            # Update session duration
            session.stats.session_duration = (
                datetime.utcnow() - session.created_at
            ).total_seconds()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating session stats {session_id}: {str(e)}")
            return False
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a session"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.status = "paused"
            logger.info(f"Paused session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing session {session_id}: {str(e)}")
            return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.status = "active"
            logger.info(f"Resumed session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming session {session_id}: {str(e)}")
            return False
    
    async def end_session(self, session_id: str, reason: str = "user_request") -> Optional[Dict]:
        """End a session and return summary"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.warning(f"Session not found for ending: {session_id}")
                return None
            
            # Update final stats
            session.status = "ended"
            session.stats.session_duration = (
                datetime.utcnow() - session.created_at
            ).total_seconds()
            
            # Create summary
            summary = await self.get_session_summary(session_id)
            summary["end_reason"] = reason
            summary["ended_at"] = datetime.utcnow().isoformat()
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Ended session: {session_id}, reason: {reason}")
            return summary
            
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {str(e)}")
            return None
    
    async def get_session_summary(self, session_id: str) -> Dict:
        """Get comprehensive session summary"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return {}
            
            # Calculate additional metrics
            detection_rate = 0.0
            if session.stats.total_frames_processed > 0:
                detection_rate = (
                    session.stats.deepfake_detections / 
                    session.stats.total_frames_processed
                ) * 100
            
            summary = {
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "status": session.status,
                "user_id": session.user_id,
                "stream_id": session.stream_id,
                "config": asdict(session.config),
                "stats": asdict(session.stats),
                "metrics": {
                    "detection_rate_percent": detection_rate,
                    "avg_processing_time": self._calculate_avg_processing_time(session),
                    "data_throughput_mbps": self._calculate_throughput(session),
                    "session_health_score": self._calculate_health_score(session)
                },
                "metadata": session.metadata
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting session summary {session_id}: {str(e)}")
            return {}
    
    def _calculate_avg_processing_time(self, session: Session) -> float:
        """Calculate average processing time per frame/chunk"""
        total_items = (session.stats.total_frames_processed + 
                      session.stats.total_audio_chunks_processed)
        if total_items > 0 and session.stats.session_duration > 0:
            return session.stats.session_duration / total_items
        return 0.0
    
    def _calculate_throughput(self, session: Session) -> float:
        """Calculate data throughput in MB/s"""
        if session.stats.session_duration > 0:
            return session.stats.data_processed_mb / session.stats.session_duration
        return 0.0
    
    def _calculate_health_score(self, session: Session) -> float:
        """Calculate session health score (0-100)"""
        try:
            score = 100.0
            
            # Deduct points for high deepfake detection rate
            if session.stats.total_frames_processed > 0:
                detection_rate = (session.stats.deepfake_detections / 
                                session.stats.total_frames_processed)
                score -= detection_rate * 50  # Max 50 points deduction
            
            # Deduct points for low processing efficiency
            avg_processing_time = self._calculate_avg_processing_time(session)
            if avg_processing_time > 1.0:  # More than 1 second per item
                score -= min(30, avg_processing_time * 10)
            
            # Deduct points for session duration issues
            if session.stats.session_duration > self.max_session_duration * 0.8:
                score -= 10
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}")
            return 50.0  # Default neutral score
    
    async def get_all_sessions(self) -> Dict[str, Dict]:
        """Get summary of all active sessions"""
        summaries = {}
        for session_id in self.active_sessions.keys():
            summaries[session_id] = await self.get_session_summary(session_id)
        return summaries
    
    async def get_session_count(self) -> Dict[str, int]:
        """Get session count statistics"""
        active_count = len(self.active_sessions)
        total_count = len(self.session_history)
        
        # Count by status
        status_counts = {}
        for session in self.active_sessions.values():
            status = session.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "active_sessions": active_count,
            "total_sessions_created": total_count,
            "status_breakdown": status_counts
        }
    
    async def cleanup_all_sessions(self):
        """Cleanup all sessions (for shutdown)"""
        try:
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.end_session(session_id, reason="system_shutdown")
            
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
            
            logger.info("All sessions cleaned up")
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {str(e)}")
    
    def get_manager_stats(self) -> Dict:
        """Get session manager statistics"""
        return {
            "active_sessions_count": len(self.active_sessions),
            "total_sessions_created": len(self.session_history),
            "cleanup_interval": self.cleanup_interval,
            "max_session_duration": self.max_session_duration,
            "inactive_timeout": self.inactive_timeout,
            "cleanup_task_running": (
                self._cleanup_task is not None and not self._cleanup_task.done()
            )
        }
