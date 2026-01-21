"""
Unified search engine for conversation analysis.

This module consolidates search functionality from multiple duplicate files
(sql_search.py, gui_sql.py, exper_sql.py) into a single, testable interface.

Usage:
    from core.search_engine import SearchEngine
    
    with SearchEngine() as search:
        results = search.keyword_search("machine learning", platform="claude")
        print(f"Found {len(results)} results")
"""

import sqlite3
import pandas as pd
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from core.database import DatabaseConnection
from core.config import Config

import logging
logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Unified conversation search engine.
    
    Provides various search methods (keyword, boolean, date range, etc.)
    with consistent interface and result formatting.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize search engine.
        
        Args:
            db_path: Path to database (default: from Config)
        """
        self.db = DatabaseConnection(db_path)
        self._connected = False
    
    def connect(self) -> None:
        """Establish database connection."""
        self.db.connect()
        self._connected = True
    
    def close(self) -> None:
        """Close database connection."""
        self.db.close()
        self._connected = False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def _ensure_connected(self) -> None:
        """Ensure database is connected."""
        if not self._connected:
            raise RuntimeError("SearchEngine not connected. Use 'with SearchEngine()' or call connect()")
    
    def keyword_search(
        self,
        query: str,
        sender: Optional[str] = None,
        platform: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Perform keyword search across messages.
        
        Args:
            query: Search keywords
            sender: Filter by sender ('human', 'assistant', or None for both)
            platform: Filter by platform ('claude', 'chatgpt', or None for all)
            limit: Maximum results to return
            
        Returns:
            DataFrame with search results
            
        Example:
            >>> with SearchEngine() as search:
            ...     results = search.keyword_search("python", platform="claude")
            ...     print(f"Found {len(results)} results")
        """
        self._ensure_connected()
        
        sql_query = """
            SELECT 
                m.id AS message_id,
                m.conversation_id,
                c.title AS conversation_title,
                c.platform,
                m.sender,
                m.content,
                m.created_at,
                m.model
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.content LIKE ?
        """
        
        params = [f"%{query}%"]
        
        if sender:
            sql_query += " AND m.sender = ?"
            params.append(sender)
        
        if platform:
            sql_query += " AND c.platform = ?"
            params.append(platform)
        
        sql_query += f" ORDER BY m.created_at DESC LIMIT {limit}"
        
        start_time = time.time()
        results = self.db.execute_query(sql_query, tuple(params))
        search_time = time.time() - start_time
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame([dict(row) for row in results])
            logger.info(f"Keyword search for '{query}' found {len(df)} results in {search_time:.2f}s")
            return df
        else:
            logger.info(f"Keyword search for '{query}' found 0 results")
            return pd.DataFrame()
    
    def date_range_search(
        self,
        start_date: str,
        end_date: str,
        platform: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Search conversations within date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            platform: Filter by platform (optional)
            limit: Maximum results to return
            
        Returns:
            DataFrame with search results
        """
        self._ensure_connected()
        
        sql_query = """
            SELECT 
                c.id AS conversation_id,
                c.title,
                c.platform,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE DATE(c.created_at) BETWEEN ? AND ?
        """
        
        params = [start_date, end_date]
        
        if platform:
            sql_query += " AND c.platform = ?"
            params.append(platform)
        
        sql_query += f" GROUP BY c.id ORDER BY c.created_at DESC LIMIT {limit}"
        
        results = self.db.execute_query(sql_query, tuple(params))
        
        if results:
            df = pd.DataFrame([dict(row) for row in results])
            logger.info(f"Date range search found {len(df)} conversations between {start_date} and {end_date}")
            return df
        else:
            return pd.DataFrame()
    
    def platform_model_search(
        self,
        platform: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Search by platform and/or model.
        
        Args:
            platform: Platform filter ('claude', 'chatgpt')
            model: Model filter (e.g., 'gpt-4', 'claude-3')
            limit: Maximum results to return
            
        Returns:
            DataFrame with search results
        """
        self._ensure_connected()
        
        sql_query = """
            SELECT 
                c.id AS conversation_id,
                c.title,
                c.platform,
                m.model,
                COUNT(m.id) AS message_count,
                c.created_at
            FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            WHERE 1=1
        """
        
        params = []
        
        if platform:
            sql_query += " AND c.platform = ?"
            params.append(platform)
        
        if model:
            sql_query += " AND m.model LIKE ?"
            params.append(f"%{model}%")
        
        sql_query += f" GROUP BY c.id, m.model ORDER BY c.created_at DESC LIMIT {limit}"
        
        results = self.db.execute_query(sql_query, tuple(params))
        
        if results:
            df = pd.DataFrame([dict(row) for row in results])
            logger.info(f"Platform/model search found {len(df)} conversations")
            return df
        else:
            return pd.DataFrame()
    
    def conversation_search(
        self,
        conversation_id: str
    ) -> Tuple[Optional[sqlite3.Row], List[sqlite3.Row]]:
        """
        Get conversation details and all messages.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            Tuple of (conversation_row, list_of_message_rows)
        """
        self._ensure_connected()
        
        conversation = self.db.get_conversation_by_id(conversation_id)
        messages = self.db.get_messages_by_conversation(conversation_id) if conversation else []
        
        return conversation, messages
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics about conversations.
        
        Returns:
            Dictionary with statistics
        """
        self._ensure_connected()
        
        stats = {}
        
        # Total conversations
        result = self.db.execute_query("SELECT COUNT(*) as count FROM conversations")
        stats['total_conversations'] = result[0]['count']
        
        # Total messages
        result = self.db.execute_query("SELECT COUNT(*) as count FROM messages")
        stats['total_messages'] = result[0]['count']
        
        # By platform
        results = self.db.execute_query("""
            SELECT platform, COUNT(*) as count 
            FROM conversations 
            GROUP BY platform
        """)
        stats['by_platform'] = {row['platform']: row['count'] for row in results}
        
        # Date range
        min_date, max_date = self.db.get_date_range()
        stats['date_range'] = {'min': min_date, 'max': max_date}
        
        # Available platforms and models
        stats['platforms'] = self.db.get_platforms()
        stats['models'] = self.db.get_models()
        
        return stats
    
    def export_results(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = 'csv'
    ) -> str:
        """
        Export search results to file.
        
        Args:
            df: DataFrame with results
            filename: Output filename (without extension)
            format: Export format ('csv' or 'json')
            
        Returns:
            Path to exported file
        """
        Config.ensure_directories()
        output_dir = Config.OUTPUT_DIR
        
        if format == 'csv':
            filepath = output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = output_dir / f"{filename}.json"
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(df)} results to {filepath}")
        return str(filepath)
