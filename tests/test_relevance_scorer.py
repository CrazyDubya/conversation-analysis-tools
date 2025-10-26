"""Tests for RelevanceScorer."""

import pytest
from pipeline.relevance_scorer import RelevanceScorer


class TestRelevanceScorer:
    """Test RelevanceScorer class."""

    def test_initialization(self, keywords):
        """Test scorer initialization."""
        scorer = RelevanceScorer(keywords=keywords)
        assert scorer.keywords == [k.lower() for k in keywords]
        assert len(scorer.stopwords) > 0

    def test_tokenize(self, keywords):
        """Test text tokenization."""
        scorer = RelevanceScorer(keywords=keywords)
        text = "Machine learning is a subset of AI that focuses on algorithms."
        tokens = scorer.tokenize(text)

        assert 'machine' in tokens
        assert 'learning' in tokens
        assert 'algorithms' in tokens
        assert 'is' not in tokens  # Stopword
        assert 'a' not in tokens   # Stopword

    def test_tokenize_empty(self, keywords):
        """Test tokenization with empty text."""
        scorer = RelevanceScorer(keywords=keywords)
        assert scorer.tokenize("") == []
        assert scorer.tokenize(None) == []

    def test_compute_tf(self, keywords):
        """Test term frequency calculation."""
        scorer = RelevanceScorer(keywords=keywords)
        tokens = ['machine', 'learning', 'machine', 'data']
        tf = scorer.compute_tf(tokens)

        assert tf['machine'] == 1.0  # Most frequent
        assert tf['learning'] == 0.5  # Half as frequent
        assert tf['data'] == 0.5

    def test_build_idf(self, keywords, sample_texts):
        """Test IDF computation."""
        scorer = RelevanceScorer(keywords=keywords)
        scorer.build_idf(sample_texts)

        assert scorer.corpus_size == len(sample_texts)
        assert len(scorer.idf_cache) > 0
        assert 'machine' in scorer.idf_cache

    def test_keyword_density_score(self, keywords):
        """Test keyword density scoring."""
        scorer = RelevanceScorer(keywords=keywords)
        text = "Machine learning and deep learning are subsets of artificial intelligence."
        score = scorer.keyword_density_score(text)

        assert 0.0 <= score <= 1.0
        assert score > 0  # Contains keywords

    def test_keyword_coverage_score(self, keywords):
        """Test keyword coverage scoring."""
        scorer = RelevanceScorer(keywords=keywords)
        text = "Machine learning and deep learning are subsets of artificial intelligence."
        score = scorer.keyword_coverage_score(text)

        assert 0.0 <= score <= 1.0
        assert score > 0  # Contains some keywords

    def test_calculate_relevance_score(self, keywords):
        """Test combined relevance scoring."""
        scorer = RelevanceScorer(keywords=keywords)
        text = "Machine learning uses neural networks for deep learning tasks."
        scores = scorer.calculate_relevance_score(text)

        assert 'density' in scores
        assert 'coverage' in scores
        assert 'tfidf' in scores
        assert 'combined' in scores
        assert 0.0 <= scores['combined'] <= 1.0

    def test_score_documents(self, keywords, sample_texts):
        """Test scoring multiple documents."""
        scorer = RelevanceScorer(keywords=keywords)
        scores = scorer.score_documents(sample_texts[:4])  # Skip duplicates

        assert len(scores) == 4
        for score in scores:
            assert 'combined' in score
            assert 0.0 <= score['combined'] <= 1.0

    def test_rank_documents(self, keywords, sample_documents):
        """Test document ranking."""
        scorer = RelevanceScorer(keywords=keywords)
        ranked = scorer.rank_documents(sample_documents[:4], top_k=2)

        assert len(ranked) == 2
        # Check sorting (highest score first)
        assert ranked[0][1] >= ranked[1][1]

    def test_rank_documents_all(self, keywords, sample_documents):
        """Test ranking without top_k limit."""
        scorer = RelevanceScorer(keywords=keywords)
        ranked = scorer.rank_documents(sample_documents[:4])

        assert len(ranked) == 4
        # Verify descending order
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i+1][1]
