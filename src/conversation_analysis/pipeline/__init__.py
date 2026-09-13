"""Content Analysis Pipeline.

A comprehensive system for analyzing scraped research content including:
- Relevance scoring
- Extractive summarization
- Duplicate detection
- Priority classification
"""

from .relevance_scorer import RelevanceScorer
from .summarizer import ExtractiveSummarizer
from .duplicate_detector import DuplicateDetector
from .priority_classifier import PriorityClassifier, PriorityLevel
from .pipeline import ContentAnalysisPipeline

__version__ = '1.0.0'

__all__ = [
    'RelevanceScorer',
    'ExtractiveSummarizer',
    'DuplicateDetector',
    'PriorityClassifier',
    'PriorityLevel',
    'ContentAnalysisPipeline',
]
