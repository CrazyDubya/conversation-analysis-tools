"""
Unified database access layer for conversation analysis tools.

This module provides a single, consistent interface for database operations,
replacing duplicate connection handling across multiple files.

Usage:
    from core.database import DatabaseConnection
    
    with DatabaseConnection() as db:
        results = db.execute_query("SELECT * FROM conversations LIMIT 10")
        for row in results:
            print(row)
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from core.config import Config

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Unified database connection manager with common query patterns.
    
    Provides consistent database access across all tools, with automatic
    connection management, error handling, and row factory configuration.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database (default: from Config)
        """
        self.db_path = db_path or Config.DB_PATH
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self._platforms_cache: Optional[List[str]] = None
        self._models_cache: Optional[List[str]] = None
        self._date_range_cache: Optional[Tuple[str, str]] = None
    
    def connect(self) -> None:
        """Establish database connection with row factory."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.debug("Database connection closed")
    
    def __enter__(self):
        """Context manager entry - establish connection."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """
        Execute SELECT query and return results.
        
        Args:
            query: SQL SELECT statement
            params: Query parameters (optional)
            
        Returns:
            List of Row objects
            
        Example:
            >>> with DatabaseConnection() as db:
            ...     results = db.execute_query(
            ...         "SELECT * FROM conversations WHERE platform = ?",
            ...         ('claude',)
            ...     )
        """
        if not self.cursor:
            raise RuntimeError("Database not connected. Use 'with DatabaseConnection()' or call connect()")
        
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Query execution error: {e}\nQuery: {query}")
            raise
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL DML statement
            params: Query parameters (optional)
            
        Returns:
            Number of affected rows
        """
        if not self.conn or not self.cursor:
            raise RuntimeError("Database not connected")
        
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            return self.cursor.rowcount
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Update execution error: {e}\nQuery: {query}")
            raise
    
    def check_view_exists(self, view_name: str) -> bool:
        """
        Check if a database view exists.
        
        Args:
            view_name: Name of the view to check
            
        Returns:
            True if view exists, False otherwise
        """
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='view' AND name=?
        """
        results = self.execute_query(query, (view_name,))
        return len(results) > 0
    
    def check_views(self, required_views: List[str] = None) -> Dict[str, bool]:
        """
        Check if required views exist in database.
        
        Args:
            required_views: List of view names to check (default: common views)
            
        Returns:
            Dictionary mapping view names to existence status
        """
        if required_views is None:
            required_views = [
                'message_pairs',
                'conversation_summary',
                'message_length_stats',
                'time_activity',
                'model_usage'
            ]
        
        status = {}
        for view_name in required_views:
            status[view_name] = self.check_view_exists(view_name)
        
        missing_views = [name for name, exists in status.items() if not exists]
        if missing_views:
            logger.warning(f"Missing database views: {', '.join(missing_views)}")
            logger.info("Run 'sqlite3 conversations.db < create_views.sql' to create views")
        
        return status
    
    def get_platforms(self) -> List[str]:
        """
        Get list of unique platforms in database.
        
        Returns:
            List of platform names
        """
        if self._platforms_cache is not None:
            return self._platforms_cache
        
        query = "SELECT DISTINCT platform FROM conversations WHERE platform IS NOT NULL"
        results = self.execute_query(query)
        self._platforms_cache = [row['platform'] for row in results]
        return self._platforms_cache
    
    def get_models(self) -> List[str]:
        """
        Get list of unique models in database.
        
        Returns:
            List of model names
        """
        if self._models_cache is not None:
            return self._models_cache
        
        query = "SELECT DISTINCT model FROM messages WHERE model IS NOT NULL"
        results = self.execute_query(query)
        self._models_cache = [row['model'] for row in results]
        return self._models_cache
    
    def get_date_range(self) -> Tuple[str, str]:
        """
        Get min and max dates from conversations.
        
        Returns:
            Tuple of (min_date, max_date) as strings
        """
        if self._date_range_cache is not None:
            return self._date_range_cache
        
        query = """
            SELECT 
                MIN(created_at) as min_date,
                MAX(created_at) as max_date
            FROM conversations
        """
        result = self.execute_query(query)[0]
        self._date_range_cache = (result['min_date'], result['max_date'])
        return self._date_range_cache
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[sqlite3.Row]:
        """
        Get conversation by ID.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            Row object or None if not found
        """
        query = "SELECT * FROM conversations WHERE id = ?"
        results = self.execute_query(query, (conversation_id,))
        return results[0] if results else None
    
    def get_messages_by_conversation(self, conversation_id: str) -> List[sqlite3.Row]:
        """
        Get all messages for a conversation.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            List of message Row objects
        """
        query = """
            SELECT * FROM messages 
            WHERE conversation_id = ? 
            ORDER BY created_at
        """
        return self.execute_query(query, (conversation_id,))


@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """
    Context manager for database connections.
    
    Args:
        db_path: Path to database (default: from Config)
        
    Yields:
        DatabaseConnection instance
        
    Example:
        >>> with get_db_connection() as db:
        ...     results = db.execute_query("SELECT * FROM conversations LIMIT 5")
    """
    db = DatabaseConnection(db_path)
    try:
        db.connect()
        yield db
    finally:
        db.close()
