"""Tests for PriorityClassifier."""

import pytest
from pipeline.priority_classifier import PriorityClassifier, PriorityLevel


class TestPriorityClassifier:
    """Test PriorityClassifier class."""

    def test_initialization(self, test_config):
        """Test classifier initialization."""
        classifier = PriorityClassifier(config=test_config['priority'])
        assert 'relevance_thresholds' in classifier.config
        assert 'weights' in classifier.config

    def test_calculate_length_score(self):
        """Test length-based scoring."""
        classifier = PriorityClassifier()
        short_text = "Short text here."
        long_text = " ".join(["word"] * 600)  # 600 words

        short_score = classifier.calculate_length_score(short_text)
        long_score = classifier.calculate_length_score(long_text)

        assert 0.0 <= short_score <= 1.0
        assert 0.0 <= long_score <= 1.0
        assert long_score > short_score

    def test_calculate_keyword_score(self):
        """Test keyword matching score."""
        classifier = PriorityClassifier()
        text_with_keywords = "This breakthrough research shows novel approaches."
        text_without = "This is just regular text without special terms."

        score1, matched1 = classifier.calculate_keyword_score(
            text_with_keywords,
            critical_keywords=['breakthrough', 'novel']
        )
        score2, matched2 = classifier.calculate_keyword_score(
            text_without,
            critical_keywords=['breakthrough', 'novel']
        )

        assert score1 > score2
        assert len(matched1) > len(matched2)

    def test_calculate_combined_score(self):
        """Test combined priority scoring."""
        classifier = PriorityClassifier()
        score = classifier.calculate_combined_score(
            relevance_score=0.8,
            length_score=0.7,
            keyword_score=0.9,
            recency_score=0.5
        )

        assert 0.0 <= score <= 1.0

    def test_score_to_priority(self, test_config):
        """Test score to priority level conversion."""
        classifier = PriorityClassifier(config=test_config['priority'])

        assert classifier.score_to_priority(0.95) == PriorityLevel.CRITICAL
        assert classifier.score_to_priority(0.70) == PriorityLevel.HIGH
        assert classifier.score_to_priority(0.50) == PriorityLevel.MEDIUM
        assert classifier.score_to_priority(0.30) == PriorityLevel.LOW
        assert classifier.score_to_priority(0.10) == PriorityLevel.NONE

    def test_classify(self):
        """Test document classification."""
        classifier = PriorityClassifier()
        text = """This is a comprehensive research paper about machine learning.
        It contains multiple paragraphs with detailed information about algorithms,
        training procedures, and evaluation metrics. The content is substantial
        and well-structured."""

        result = classifier.classify(text, relevance_score=0.8)

        assert isinstance(result.level, PriorityLevel)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reasons, list)
        assert len(result.reasons) > 0
        assert isinstance(result.metadata, dict)

    def test_classify_batch(self):
        """Test batch classification."""
        classifier = PriorityClassifier()
        documents = [
            (0, "Short text.", 0.5),
            (1, " ".join(["word"] * 300), 0.8),  # Long text, high relevance
            (2, "Medium length text with some content here.", 0.6),
        ]

        results = classifier.classify_batch(documents)

        assert len(results) == 3
        assert all(isinstance(r[1].level, PriorityLevel) for r in results)

    def test_rank_by_priority(self):
        """Test ranking by priority."""
        classifier = PriorityClassifier()
        documents = [
            (0, "Short text.", 0.3),
            (1, " ".join(["word"] * 300), 0.9),
            (2, "Medium text.", 0.6),
        ]

        ranked = classifier.rank_by_priority(documents)

        assert len(ranked) == 3
        # Should be sorted by score (descending)
        for i in range(len(ranked) - 1):
            assert ranked[i][1].score >= ranked[i+1][1].score

    def test_rank_by_priority_filtered(self):
        """Test ranking with priority level filter."""
        classifier = PriorityClassifier()
        documents = [
            (0, "Short.", 0.2),
            (1, " ".join(["word"] * 300), 0.9),
            (2, "Medium text here.", 0.6),
        ]

        ranked = classifier.rank_by_priority(documents, filter_level=PriorityLevel.MEDIUM)

        # Should only include MEDIUM or higher
        assert all(r[1].level.value >= PriorityLevel.MEDIUM.value for r in ranked)

    def test_get_priority_distribution(self):
        """Test priority distribution calculation."""
        classifier = PriorityClassifier()
        documents = [
            (0, "Short.", 0.2),
            (1, " ".join(["word"] * 300), 0.9),
            (2, "Medium text.", 0.6),
            (3, "Another short.", 0.3),
        ]

        distribution = classifier.get_priority_distribution(documents)

        assert isinstance(distribution, dict)
        assert sum(distribution.values()) == len(documents)
        assert all(level in distribution for level in PriorityLevel)
