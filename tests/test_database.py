"""
Tests for core.database module.

Tests database connection management, context managers,
and common query patterns.
"""

import pytest
import sqlite3
from pathlib import Path
from core.database import DatabaseConnection, get_db_connection


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create test tables
    cursor.execute("""
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            platform TEXT,
            created_at TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            content TEXT,
            sender TEXT,
            model TEXT,
            created_at TEXT
        )
    """)
    
    # Insert test data
    cursor.execute("""
        INSERT INTO conversations VALUES
        ('conv1', 'Test Conversation', 'claude', '2024-01-01'),
        ('conv2', 'Another Test', 'chatgpt', '2024-01-02')
    """)
    
    cursor.execute("""
        INSERT INTO messages VALUES
        ('msg1', 'conv1', 'Hello', 'human', 'claude-3', '2024-01-01'),
        ('msg2', 'conv1', 'Hi there', 'assistant', 'claude-3', '2024-01-01'),
        ('msg3', 'conv2', 'Test', 'human', 'gpt-4', '2024-01-02')
    """)
    
    conn.commit()
    conn.close()
    
    return str(db_path)


def test_database_connection_init(temp_db):
    """Test database connection initialization."""
    db = DatabaseConnection(temp_db)
    assert db.db_path == temp_db
    assert db.conn is None
    assert db.cursor is None


def test_database_connect_close(temp_db):
    """Test connection and closing."""
    db = DatabaseConnection(temp_db)
    
    # Connect
    db.connect()
    assert db.conn is not None
    assert db.cursor is not None
    
    # Close
    db.close()
    assert db.conn is None
    assert db.cursor is None


def test_context_manager(temp_db):
    """Test context manager usage."""
    with DatabaseConnection(temp_db) as db:
        assert db.conn is not None
        assert db.cursor is not None
        results = db.execute_query("SELECT COUNT(*) as count FROM conversations")
        assert len(results) == 1
        assert results[0]['count'] == 2


def test_execute_query(temp_db):
    """Test query execution."""
    with DatabaseConnection(temp_db) as db:
        results = db.execute_query("SELECT * FROM conversations WHERE platform = ?", ('claude',))
        assert len(results) == 1
        assert results[0]['platform'] == 'claude'


def test_get_platforms(temp_db):
    """Test get_platforms method."""
    with DatabaseConnection(temp_db) as db:
        platforms = db.get_platforms()
        assert len(platforms) == 2
        assert 'claude' in platforms
        assert 'chatgpt' in platforms


def test_get_models(temp_db):
    """Test get_models method."""
    with DatabaseConnection(temp_db) as db:
        models = db.get_models()
        assert len(models) == 2
        assert 'claude-3' in models or 'gpt-4' in models


def test_get_date_range(temp_db):
    """Test get_date_range method."""
    with DatabaseConnection(temp_db) as db:
        min_date, max_date = db.get_date_range()
        assert min_date == '2024-01-01'
        assert max_date == '2024-01-02'


def test_get_conversation_by_id(temp_db):
    """Test get_conversation_by_id method."""
    with DatabaseConnection(temp_db) as db:
        conv = db.get_conversation_by_id('conv1')
        assert conv is not None
        assert conv['id'] == 'conv1'
        assert conv['title'] == 'Test Conversation'
        
        # Non-existent conversation
        conv = db.get_conversation_by_id('nonexistent')
        assert conv is None


def test_get_messages_by_conversation(temp_db):
    """Test get_messages_by_conversation method."""
    with DatabaseConnection(temp_db) as db:
        messages = db.get_messages_by_conversation('conv1')
        assert len(messages) == 2
        assert messages[0]['conversation_id'] == 'conv1'


def test_get_db_connection_context_manager(temp_db):
    """Test get_db_connection helper function."""
    with get_db_connection(temp_db) as db:
        assert db.conn is not None
        results = db.execute_query("SELECT COUNT(*) as count FROM messages")
        assert results[0]['count'] == 3
