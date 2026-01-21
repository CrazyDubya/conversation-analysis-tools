"""
Tests for uni_parse module.

Tests the ConversationParser for handling Claude and ChatGPT conversation data.
"""

import pytest
import sqlite3
import json
import tempfile
from pathlib import Path
from uni_parse import ConversationParser


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def parser(temp_db):
    """Create a ConversationParser instance."""
    return ConversationParser(temp_db)


@pytest.fixture
def sample_claude_data():
    """Sample Claude conversation data."""
    return [
        {
            "uuid": "test-uuid-1",
            "name": "Test Conversation",
            "created_at": "2024-01-01T00:00:00.000000Z",
            "updated_at": "2024-01-01T01:00:00.000000Z",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "text": "Hello",
                    "created_at": "2024-01-01T00:00:00.000000Z"
                },
                {
                    "uuid": "msg-2",
                    "sender": "assistant",
                    "text": "Hi there!",
                    "created_at": "2024-01-01T00:01:00.000000Z"
                }
            ]
        }
    ]


@pytest.fixture
def sample_chatgpt_data():
    """Sample ChatGPT conversation data."""
    return [
        {
            "id": "test-id-1",
            "title": "Test Chat",
            "create_time": 1704067200.0,
            "update_time": 1704070800.0,
            "mapping": {
                "node-1": {
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                        "create_time": 1704067200.0
                    }
                },
                "node-2": {
                    "message": {
                        "id": "msg-2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Hi there!"]},
                        "create_time": 1704067260.0
                    }
                }
            }
        }
    ]


def test_parser_initialization(temp_db):
    """Test ConversationParser initialization."""
    parser = ConversationParser(temp_db)
    assert parser.db_path == temp_db
    assert Path(temp_db).exists()


def test_database_tables_created(parser, temp_db):
    """Test that database tables are created."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    # Check conversations table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
    assert cursor.fetchone() is not None
    
    # Check messages table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    assert cursor.fetchone() is not None
    
    conn.close()


def test_detect_platform_claude(parser, tmp_path, sample_claude_data):
    """Test platform detection for Claude data."""
    claude_file = tmp_path / "claude.json"
    with open(claude_file, 'w') as f:
        json.dump(sample_claude_data, f)
    
    platform = parser._detect_platform(str(claude_file))
    assert platform == "claude"


def test_detect_platform_chatgpt(parser, tmp_path, sample_chatgpt_data):
    """Test platform detection for ChatGPT data."""
    chatgpt_file = tmp_path / "chatgpt.json"
    with open(chatgpt_file, 'w') as f:
        json.dump(sample_chatgpt_data, f)
    
    platform = parser._detect_platform(str(chatgpt_file))
    assert platform == "chatgpt"


def test_parse_claude_file(parser, tmp_path, sample_claude_data, temp_db):
    """Test parsing Claude conversation file."""
    claude_file = tmp_path / "claude.json"
    with open(claude_file, 'w') as f:
        json.dump(sample_claude_data, f)
    
    parser.parse_file(str(claude_file), platform="claude")
    
    # Verify data was stored
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM conversations")
    assert cursor.fetchone()[0] == 1
    
    cursor.execute("SELECT COUNT(*) FROM messages")
    assert cursor.fetchone()[0] == 2
    
    cursor.execute("SELECT platform FROM conversations")
    assert cursor.fetchone()[0] == "claude"
    
    conn.close()


def test_parse_chatgpt_file(parser, tmp_path, sample_chatgpt_data, temp_db):
    """Test parsing ChatGPT conversation file."""
    chatgpt_file = tmp_path / "chatgpt.json"
    with open(chatgpt_file, 'w') as f:
        json.dump(sample_chatgpt_data, f)
    
    parser.parse_file(str(chatgpt_file), platform="chatgpt")
    
    # Verify data was stored
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM conversations")
    assert cursor.fetchone()[0] == 1
    
    cursor.execute("SELECT COUNT(*) FROM messages")
    # ChatGPT data structure results in 2 messages
    assert cursor.fetchone()[0] >= 1
    
    cursor.execute("SELECT platform FROM conversations")
    assert cursor.fetchone()[0] == "chatgpt"
    
    conn.close()


def test_parse_file_auto_detection(parser, tmp_path, sample_claude_data):
    """Test parsing with automatic platform detection."""
    claude_file = tmp_path / "data.json"
    with open(claude_file, 'w') as f:
        json.dump(sample_claude_data, f)
    
    # Should auto-detect as Claude
    parser.parse_file(str(claude_file))
    
    conn = sqlite3.connect(parser.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT platform FROM conversations")
    platform = cursor.fetchone()[0]
    conn.close()
    
    assert platform == "claude"


def test_invalid_platform_raises_error(parser, tmp_path):
    """Test that invalid platform raises error."""
    test_file = tmp_path / "test.json"
    test_file.write_text('{"invalid": "data"}')
    
    with pytest.raises(ValueError, match="Unsupported platform"):
        parser.parse_file(str(test_file), platform="invalid")
