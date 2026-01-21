"""
Centralized configuration management for conversation analysis tools.

This module provides a single source of truth for all configuration settings,
replacing hardcoded paths and magic numbers throughout the codebase.

Usage:
    from core.config import Config
    
    config = Config()
    db_path = config.DB_PATH
    colors = config.PLATFORM_COLORS
    
    # Or load from environment file
    config = Config.load_from_env('.env')
"""

import os
from pathlib import Path
from typing import Optional, Dict


class Config:
    """Centralized configuration management."""
    
    # Database configuration
    DB_PATH: str = os.getenv('DB_PATH', 'conversations.db')
    
    # Directory configuration
    OUTPUT_DIR: Path = Path(os.getenv('OUTPUT_DIR', 'output'))
    VISUALIZATIONS_DIR: Path = Path(os.getenv('VISUALIZATIONS_DIR', 'visualizations'))
    ADVANCED_VISUALIZATIONS_DIR: Path = Path(os.getenv('ADVANCED_VISUALIZATIONS_DIR', 'advanced_visualizations'))
    CONTENT_ANALYSIS_DIR: Path = Path(os.getenv('CONTENT_ANALYSIS_DIR', 'content_analysis'))
    LOGS_DIR: Path = Path(os.getenv('LOGS_DIR', 'logs'))
    
    # Pipeline configuration
    PIPELINE_CONFIG: str = os.getenv('PIPELINE_CONFIG', 'config/pipeline_config.yaml')
    MAX_RESULTS: int = int(os.getenv('MAX_RESULTS', '100'))
    DEFAULT_PLATFORM: str = os.getenv('DEFAULT_PLATFORM', 'claude')
    
    # Logging configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # UI configuration - Platform colors for visualizations
    PLATFORM_COLORS: Dict[str, str] = {
        'claude': '#8C52FF',      # Purple
        'chatgpt': '#00A67E',     # Green
        'unknown': '#808080'      # Gray
    }
    
    # Matplotlib style settings
    PLOT_STYLE: str = 'fivethirtyeight'
    FIGURE_SIZE: tuple = (15, 10)
    FIGURE_DPI: int = 100
    FONT_SIZE: int = 12
    
    @classmethod
    def load_from_env(cls, env_file: Optional[str] = '.env') -> 'Config':
        """
        Load configuration from .env file.
        
        Args:
            env_file: Path to .env file (default: '.env')
            
        Returns:
            Config instance with settings loaded from environment
            
        Example:
            >>> config = Config.load_from_env('.env.local')
            >>> print(config.DB_PATH)
        """
        if env_file and Path(env_file).exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
            except ImportError:
                print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
        return cls()
    
    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create all configured directories if they don't exist.
        
        Example:
            >>> Config.ensure_directories()
        """
        directories = [
            cls.OUTPUT_DIR,
            cls.VISUALIZATIONS_DIR,
            cls.ADVANCED_VISUALIZATIONS_DIR,
            cls.CONTENT_ANALYSIS_DIR,
            cls.LOGS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_db_path(cls) -> str:
        """
        Get the database path, checking if file exists.
        
        Returns:
            Path to database file
            
        Raises:
            FileNotFoundError: If database file doesn't exist and DB_PATH is not default
        """
        db_path = cls.DB_PATH
        if not os.path.exists(db_path) and db_path != 'conversations.db':
            raise FileNotFoundError(
                f"Database file not found: {db_path}\n"
                f"Set DB_PATH environment variable or create .env file"
            )
        return db_path


# Create a singleton instance for easy import
config = Config()
