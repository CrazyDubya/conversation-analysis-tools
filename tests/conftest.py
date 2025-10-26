"""Pytest configuration and fixtures."""

import pytest
import tempfile
import sqlite3
import os


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns in large datasets.",
        "Natural language processing enables computers to understand and generate human language.",
        "The transformer architecture revolutionized NLP by introducing attention mechanisms.",
        "This is a short text.",
        "Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data.",  # Duplicate
    ]


@pytest.fixture
def sample_documents(sample_texts):
    """Sample documents with IDs."""
    return [(i, text) for i, text in enumerate(sample_texts)]


@pytest.fixture
def keywords():
    """Sample keywords for testing."""
    return [
        'machine learning',
        'deep learning',
        'neural network',
        'artificial intelligence',
        'natural language processing',
        'transformer',
        'attention'
    ]


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    # Create temp database
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # Create schema
    cursor.execute("""
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY,
            platform TEXT,
            title TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            sender TEXT,
            content TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)

    # Insert test data
    cursor.execute("INSERT INTO conversations VALUES (1, 'claude', 'Test Conversation')")
    cursor.execute("INSERT INTO conversations VALUES (2, 'chatgpt', 'Another Test')")

    messages = [
        (1, 1, 'assistant', 'Machine learning is a subset of artificial intelligence.'),
        (2, 1, 'assistant', 'Deep learning uses neural networks with multiple layers.'),
        (3, 2, 'assistant', 'Natural language processing enables computers to understand language.'),
        (4, 2, 'user', 'Tell me about transformers.'),
        (5, 2, 'assistant', 'The transformer architecture revolutionized NLP.'),
    ]

    cursor.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", messages)
    conn.commit()
    conn.close()

    yield path

    # Cleanup
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        'keywords': [
            'machine learning',
            'deep learning',
            'neural network',
            'artificial intelligence',
        ],
        'relevance': {
            'weights': {
                'density': 0.3,
                'coverage': 0.4,
                'tfidf': 0.3
            }
        },
        'summarizer': {
            'damping': 0.85,
        },
        'summary_sentences': 2,
        'duplicate_threshold': 0.8,
        'priority': {
            'relevance_thresholds': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            },
            'length_thresholds': {
                'min_words': 10,
                'substantial': 50,
                'comprehensive': 100
            },
            'keyword_thresholds': {
                'critical_keywords': ['breakthrough', 'novel'],
                'high_keywords': ['important', 'significant'],
                'urgent_patterns': []
            },
            'weights': {
                'relevance': 0.4,
                'length': 0.2,
                'keyword_match': 0.3,
                'recency': 0.1
            },
            'priority_thresholds': {
                'critical': 0.85,
                'high': 0.65,
                'medium': 0.45,
                'low': 0.25
            }
        }
    }
