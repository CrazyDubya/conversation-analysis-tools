"""
Tests for core.search_engine module.

Tests search functionality, result formatting, and export capabilities.
"""

import pytest
import pandas as pd
from pathlib import Path
from core.search_engine import SearchEngine
from tests.test_database import temp_db  # Reuse the fixture


def test_search_engine_init(temp_db):
    """Test search engine initialization."""
    search = SearchEngine(temp_db)
    assert search.db.db_path == temp_db
    assert not search._connected


def test_search_engine_connect_close(temp_db):
    """Test connection and closing."""
    search = SearchEngine(temp_db)
    
    search.connect()
    assert search._connected
    
    search.close()
    assert not search._connected


def test_context_manager(temp_db):
    """Test context manager usage."""
    with SearchEngine(temp_db) as search:
        assert search._connected
        stats = search.get_conversation_stats()
        assert stats['total_conversations'] == 2


def test_keyword_search(temp_db):
    """Test keyword search functionality."""
    with SearchEngine(temp_db) as search:
        # Search for existing content
        results = search.keyword_search("Hello")
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'message_id' in results.columns
        assert 'content' in results.columns
        
        # Search with no results
        results = search.keyword_search("nonexistent")
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0


def test_keyword_search_with_filters(temp_db):
    """Test keyword search with platform filter."""
    with SearchEngine(temp_db) as search:
        # Filter by platform
        results = search.keyword_search("Test", platform="chatgpt")
        assert isinstance(results, pd.DataFrame)
        if len(results) > 0:
            assert all(results['platform'] == 'chatgpt')


def test_date_range_search(temp_db):
    """Test date range search."""
    with SearchEngine(temp_db) as search:
        results = search.date_range_search("2024-01-01", "2024-01-02")
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2


def test_platform_model_search(temp_db):
    """Test platform/model search."""
    with SearchEngine(temp_db) as search:
        # Search by platform
        results = search.platform_model_search(platform="claude")
        assert isinstance(results, pd.DataFrame)
        if len(results) > 0:
            assert all(results['platform'] == 'claude')


def test_conversation_search(temp_db):
    """Test conversation search by ID."""
    with SearchEngine(temp_db) as search:
        conversation, messages = search.conversation_search('conv1')
        
        assert conversation is not None
        assert conversation['id'] == 'conv1'
        assert len(messages) == 2


def test_get_conversation_stats(temp_db):
    """Test conversation statistics."""
    with SearchEngine(temp_db) as search:
        stats = search.get_conversation_stats()
        
        assert 'total_conversations' in stats
        assert 'total_messages' in stats
        assert 'by_platform' in stats
        assert 'date_range' in stats
        assert 'platforms' in stats
        assert 'models' in stats
        
        assert stats['total_conversations'] == 2
        assert stats['total_messages'] == 3


def test_export_results(temp_db, tmp_path):
    """Test exporting search results."""
    from core.config import Config
    
    # Override output directory
    original_output = Config.OUTPUT_DIR
    Config.OUTPUT_DIR = tmp_path
    
    try:
        with SearchEngine(temp_db) as search:
            results = search.keyword_search("Hello")
            
            if len(results) > 0:
                # Export as CSV
                filepath = search.export_results(results, "test_results", format='csv')
                assert Path(filepath).exists()
                
                # Export as JSON
                filepath = search.export_results(results, "test_results_json", format='json')
                assert Path(filepath).exists()
    finally:
        Config.OUTPUT_DIR = original_output


def test_ensure_connected_raises_error(temp_db):
    """Test that operations without connection raise error."""
    search = SearchEngine(temp_db)
    
    with pytest.raises(RuntimeError, match="not connected"):
        search.keyword_search("test")
