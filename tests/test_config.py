"""
Tests for core.config module.

Tests configuration management, environment variable handling,
and directory creation utilities.
"""

import pytest
import os
from pathlib import Path
from core.config import Config


def test_config_defaults():
    """Test default configuration values."""
    config = Config()
    
    # Check defaults
    assert config.DB_PATH == os.getenv('DB_PATH', 'conversations.db')
    assert isinstance(config.OUTPUT_DIR, Path)
    assert config.DEFAULT_PLATFORM == os.getenv('DEFAULT_PLATFORM', 'claude')
    assert config.LOG_LEVEL == os.getenv('LOG_LEVEL', 'INFO')


def test_platform_colors():
    """Test platform color configuration."""
    config = Config()
    
    assert 'claude' in config.PLATFORM_COLORS
    assert 'chatgpt' in config.PLATFORM_COLORS
    assert isinstance(config.PLATFORM_COLORS['claude'], str)
    assert config.PLATFORM_COLORS['claude'].startswith('#')


def test_matplotlib_settings():
    """Test matplotlib configuration."""
    config = Config()
    
    assert config.PLOT_STYLE == 'fivethirtyeight'
    assert isinstance(config.FIGURE_SIZE, tuple)
    assert len(config.FIGURE_SIZE) == 2
    assert config.FIGURE_DPI > 0
    assert config.FONT_SIZE > 0


def test_ensure_directories(tmp_path):
    """Test directory creation."""
    # Temporarily override paths to use tmp_path
    original_output = Config.OUTPUT_DIR
    Config.OUTPUT_DIR = tmp_path / "output"
    Config.VISUALIZATIONS_DIR = tmp_path / "viz"
    Config.LOGS_DIR = tmp_path / "logs"
    
    try:
        Config.ensure_directories()
        
        assert Config.OUTPUT_DIR.exists()
        assert Config.VISUALIZATIONS_DIR.exists()
        assert Config.LOGS_DIR.exists()
    finally:
        # Restore original
        Config.OUTPUT_DIR = original_output


def test_singleton_instance():
    """Test that config singleton exists."""
    from core.config import config
    
    assert config is not None
    assert isinstance(config, Config)
    assert config.DB_PATH is not None
