"""
Database Manager - Handles data persistence and analytics
"""

import os
import json
import asyncio
import aiosqlite
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger


class DatabaseManager:
    """Manages database operations for session logging and analytics"""
    
    def __init__(self, db_path: str = "data/trueface.db"):
        self.db_path = db_path
        self.db_dir = os.path.dirname(db_path)
        self.connection_pool = None
        self.max_connections = 10
        self._initialized = False
    
    async def initialize(self):
        """Initialize database and create tables"""
        try:
            # Create data directory
            os.makedirs(self.db_dir, exist_ok=True)
            
            # Create tables
            await self._create_tables()
            
            self._initialized = True
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Sessions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    stream_id TEXT,
                    created_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    status TEXT,
                    config TEXT,
                    stats TEXT,
                    metadata TEXT
                )
            """)
            
            # Analysis results table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    analysis_type TEXT,
                    timestamp TIMESTAMP,
                    results TEXT,
                    deepfake_score REAL,
                    confidence REAL,
                    is_deepfake BOOLEAN,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Detection events table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT,
                    severity TEXT,
                    description TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # System logs table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT,
                    message TEXT,
                    component TEXT,
                    session_id TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Analytics aggregations table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS analytics_daily (
                    date DATE PRIMARY KEY,
                    total_sessions INTEGER,
                    total_analyses INTEGER,
                    total_deepfakes_detected INTEGER,
                    avg_deepfake_score REAL,
                    avg_processing_time REAL,
                    total_data_processed_mb REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_analysis_session_id ON analysis_results(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_results(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_detection_session_id ON detection_events(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)")
            
            await db.commit()
            logger.info("Database tables created successfully")
    
    async def store_session(self, session_data: Dict) -> bool:
        """Store session information"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, user_id, stream_id, created_at, ended_at, status, config, stats, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_data.get("session_id"),
                    session_data.get("user_id"),
                    session_data.get("stream_id"),
                    session_data.get("created_at"),
                    session_data.get("ended_at"),
                    session_data.get("status"),
                    json.dumps(session_data.get("config", {})),
                    json.dumps(session_data.get("stats", {})),
                    json.dumps(session_data.get("metadata", {}))
                ))
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error storing session: {str(e)}")
            return False
    
    async def store_analysis_result(self, session_id: str, analysis_type: str, 
                                  results: Dict, timestamp: str) -> bool:
        """Store analysis results"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO analysis_results 
                    (session_id, analysis_type, timestamp, results, deepfake_score, 
                     confidence, is_deepfake, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    analysis_type,
                    timestamp,
                    json.dumps(results),
                    results.get("overall_score", 0.0),
                    results.get("confidence", 0.0),
                    results.get("is_deepfake", False),
                    results.get("processing_time", 0.0)
                ))
                await db.commit()
                
                # Update daily analytics
                await self._update_daily_analytics(results)
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")
            return False
    
    async def store_detection_event(self, session_id: str, event_type: str, 
                                  severity: str, description: str, 
                                  metadata: Dict = None) -> bool:
        """Store detection events"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO detection_events 
                    (session_id, event_type, severity, description, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    event_type,
                    severity,
                    description,
                    json.dumps(metadata or {}),
                    datetime.utcnow().isoformat()
                ))
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error storing detection event: {str(e)}")
            return False
    
    async def store_system_log(self, level: str, message: str, component: str, 
                             session_id: str = None, metadata: Dict = None) -> bool:
        """Store system logs"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO system_logs 
                    (level, message, component, session_id, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    level,
                    message,
                    component,
                    session_id,
                    json.dumps(metadata or {}),
                    datetime.utcnow().isoformat()
                ))
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error storing system log: {str(e)}")
            return False
    
    async def get_session_logs(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Get analysis logs for a specific session"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute("""
                    SELECT * FROM analysis_results 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, limit))
                
                rows = await cursor.fetchall()
                
                logs = []
                for row in rows:
                    log_entry = dict(row)
                    # Parse JSON fields
                    log_entry["results"] = json.loads(log_entry["results"])
                    logs.append(log_entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Error retrieving session logs: {str(e)}")
            return []
    
    async def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute("""
                    SELECT * FROM sessions WHERE session_id = ?
                """, (session_id,))
                
                row = await cursor.fetchone()
                
                if row:
                    session_info = dict(row)
                    # Parse JSON fields
                    session_info["config"] = json.loads(session_info["config"] or "{}")
                    session_info["stats"] = json.loads(session_info["stats"] or "{}")
                    session_info["metadata"] = json.loads(session_info["metadata"] or "{}")
                    return session_info
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving session info: {str(e)}")
            return None
    
    async def get_detection_events(self, session_id: str = None, 
                                 event_type: str = None, 
                                 severity: str = None,
                                 limit: int = 100) -> List[Dict]:
        """Get detection events with optional filters"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                query = "SELECT * FROM detection_events WHERE 1=1"
                params = []
                
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                
                if severity:
                    query += " AND severity = ?"
                    params.append(severity)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                events = []
                for row in rows:
                    event = dict(row)
                    event["metadata"] = json.loads(event["metadata"])
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error(f"Error retrieving detection events: {str(e)}")
            return []
    
    async def get_analytics_summary(self, days: int = 7) -> Dict:
        """Get analytics summary for the specified number of days"""
        try:
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Get daily analytics
                cursor = await db.execute("""
                    SELECT * FROM analytics_daily 
                    WHERE date BETWEEN ? AND ? 
                    ORDER BY date DESC
                """, (start_date.isoformat(), end_date.isoformat()))
                
                daily_data = [dict(row) for row in await cursor.fetchall()]
                
                # Get overall statistics
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(DISTINCT session_id) as unique_sessions
                    FROM analysis_results 
                    WHERE DATE(timestamp) BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                
                overall_stats = dict(await cursor.fetchone())
                
                # Get deepfake detection statistics
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_analyses,
                        SUM(CASE WHEN is_deepfake THEN 1 ELSE 0 END) as deepfake_detections,
                        AVG(deepfake_score) as avg_deepfake_score,
                        AVG(confidence) as avg_confidence,
                        AVG(processing_time) as avg_processing_time
                    FROM analysis_results 
                    WHERE DATE(timestamp) BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))
                
                detection_stats = dict(await cursor.fetchone())
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days
                    },
                    "daily_data": daily_data,
                    "overall_stats": overall_stats,
                    "detection_stats": detection_stats
                }
                
        except Exception as e:
            logger.error(f"Error retrieving analytics summary: {str(e)}")
            return {}
    
    async def _update_daily_analytics(self, results: Dict):
        """Update daily analytics aggregations"""
        try:
            today = datetime.utcnow().date()
            
            async with aiosqlite.connect(self.db_path) as db:
                # Check if record exists for today
                cursor = await db.execute("""
                    SELECT * FROM analytics_daily WHERE date = ?
                """, (today.isoformat(),))
                
                existing = await cursor.fetchone()
                
                if existing:
                    # Update existing record
                    await db.execute("""
                        UPDATE analytics_daily SET
                            total_analyses = total_analyses + 1,
                            total_deepfakes_detected = total_deepfakes_detected + ?,
                            avg_deepfake_score = (avg_deepfake_score + ?) / 2,
                            avg_processing_time = (avg_processing_time + ?) / 2,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE date = ?
                    """, (
                        1 if results.get("is_deepfake", False) else 0,
                        results.get("overall_score", 0.0),
                        results.get("processing_time", 0.0),
                        today.isoformat()
                    ))
                else:
                    # Create new record
                    await db.execute("""
                        INSERT INTO analytics_daily 
                        (date, total_sessions, total_analyses, total_deepfakes_detected,
                         avg_deepfake_score, avg_processing_time, total_data_processed_mb)
                        VALUES (?, 1, 1, ?, ?, ?, 0)
                    """, (
                        today.isoformat(),
                        1 if results.get("is_deepfake", False) else 0,
                        results.get("overall_score", 0.0),
                        results.get("processing_time", 0.0)
                    ))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error updating daily analytics: {str(e)}")
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data beyond retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Clean up old analysis results
                await db.execute("""
                    DELETE FROM analysis_results 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
                
                # Clean up old detection events
                await db.execute("""
                    DELETE FROM detection_events 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
                
                # Clean up old system logs
                await db.execute("""
                    DELETE FROM system_logs 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
                
                await db.commit()
                
                logger.info(f"Cleaned up data older than {days_to_keep} days")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                stats = {}
                
                # Table row counts
                tables = ["sessions", "analysis_results", "detection_events", "system_logs", "analytics_daily"]
                
                for table in tables:
                    cursor = await db.execute(f"SELECT COUNT(*) as count FROM {table}")
                    result = await cursor.fetchone()
                    stats[f"{table}_count"] = result["count"]
                
                # Database file size
                if os.path.exists(self.db_path):
                    stats["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    async def close(self):
        """Close database connections"""
        try:
            # Perform any cleanup operations
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {str(e)}")
