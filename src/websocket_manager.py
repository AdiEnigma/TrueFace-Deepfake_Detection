"""
WebSocket Manager - Handles WebSocket connections and messaging
"""

import json
from typing import Dict, List
from fastapi import WebSocket
from loguru import logger


class WebSocketManager:
    """Manages WebSocket connections and messaging"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_metadata[session_id] = {
            "connected_at": None,
            "last_ping": None,
            "message_count": 0
        }
        logger.info(f"WebSocket connection established: {session_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        session_id = None
        for sid, ws in self.active_connections.items():
            if ws == websocket:
                session_id = sid
                break
        
        if session_id:
            del self.active_connections[session_id]
            if session_id in self.connection_metadata:
                del self.connection_metadata[session_id]
            logger.info(f"WebSocket connection closed: {session_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
    
    async def send_to_session(self, message: dict, session_id: str):
        """Send a message to a specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await self.send_personal_message(message, websocket)
        else:
            logger.warning(f"Session not found: {session_id}")
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients"""
        disconnected_sessions = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                await self.send_personal_message(message, websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {str(e)}")
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            if session_id in self.connection_metadata:
                del self.connection_metadata[session_id]
    
    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_connections.keys())
    
    async def ping_all_connections(self):
        """Send ping to all connections to keep them alive"""
        ping_message = {
            "type": "ping",
            "timestamp": None
        }
        await self.broadcast(ping_message)
