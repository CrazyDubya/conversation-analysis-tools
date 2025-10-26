"""Tests for ExtractiveSummarizer."""

import pytest
from pipeline.summarizer import ExtractiveSummarizer


class TestExtractiveSummarizer:
    """Test ExtractiveSummarizer class."""

    def test_initialization(self):
        """Test summarizer initialization."""
        summarizer = ExtractiveSummarizer(damping=0.85)
        assert summarizer.damping == 0.85
        assert len(summarizer.stopwords) > 0

    def test_split_sentences(self):
        """Test sentence splitting."""
        summarizer = ExtractiveSummarizer()
        text = "This is sentence one. This is sentence two! And this is sentence three?"
        sentences = summarizer.split_sentences(text)

        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]

    def test_split_sentences_short(self):
        """Test filtering of very short sentences."""
        summarizer = ExtractiveSummarizer()
        text = "Hi. This is a longer sentence with more words. Ok."
        sentences = summarizer.split_sentences(text)

        # Should filter out "Hi" and "Ok" (too short)
        assert len(sentences) == 1

    def test_tokenize(self):
        """Test tokenization."""
        summarizer = ExtractiveSummarizer()
        text = "Machine learning is a subset of artificial intelligence."
        tokens = summarizer.tokenize(text)

        assert 'machine' in tokens
        assert 'learning' in tokens
        assert 'is' not in tokens  # Stopword

    def test_sentence_similarity(self):
        """Test sentence similarity calculation."""
        summarizer = ExtractiveSummarizer()
        sent1 = "Machine learning is important for AI."
        sent2 = "Machine learning is crucial for artificial intelligence."
        sent3 = "Dogs and cats are popular pets."

        sim_high = summarizer.sentence_similarity(sent1, sent2)
        sim_low = summarizer.sentence_similarity(sent1, sent3)

        assert 0.0 <= sim_high <= 1.0
        assert 0.0 <= sim_low <= 1.0
        assert sim_high > sim_low  # More similar sentences

    def test_build_similarity_matrix(self):
        """Test similarity matrix construction."""
        summarizer = ExtractiveSummarizer()
        sentences = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "NLP enables language understanding."
        ]
        matrix = summarizer.build_similarity_matrix(sentences)

        assert matrix.shape == (3, 3)
        assert matrix[0][0] == 0  # Diagonal is zero (no self-similarity)
        assert matrix[0][1] == matrix[1][0]  # Symmetric

    def test_summarize_short_text(self):
        """Test summarization of short text (fewer sentences than requested)."""
        summarizer = ExtractiveSummarizer()
        text = "This is a short text. It has only two sentences."
        summary = summarizer.summarize(text, num_sentences=5)

        # Should return all sentences when text is short
        assert len(summary) <= 2

    def test_summarize_long_text(self):
        """Test summarization of longer text."""
        summarizer = ExtractiveSummarizer()
        text = """Machine learning is a method of data analysis that automates analytical model building.
        It is a branch of artificial intelligence based on the idea that systems can learn from data.
        Deep learning is part of a broader family of machine learning methods based on artificial neural networks.
        Natural language processing strives to build machines that understand and respond to text or voice data.
        The field combines linguistics and computer science to make human language understandable to computers.
        """
        summary = summarizer.summarize(text, num_sentences=2)

        assert len(summary) == 2
        assert all(isinstance(s, str) for s in summary)

    def test_summarize_with_scores(self):
        """Test summarization returning scores."""
        summarizer = ExtractiveSummarizer()
        text = """Machine learning is important. Deep learning is powerful.
        Natural language processing is useful. AI transforms industries.
        Data science drives decisions."""
        summary, scores = summarizer.summarize(text, num_sentences=2, return_scores=True)

        assert len(summary) == 2
        assert len(scores) == 2
        assert all(0 <= s <= 1 for s in scores)

    def test_summarize_to_text(self):
        """Test text-based summary output."""
        summarizer = ExtractiveSummarizer()
        text = """Machine learning is powerful. Deep learning is advanced.
        Natural language processing is important."""
        summary_text = summarizer.summarize_to_text(text, num_sentences=2)

        assert isinstance(summary_text, str)
        assert len(summary_text) > 0

    def test_summarize_multiple(self):
        """Test summarizing multiple documents."""
        summarizer = ExtractiveSummarizer()
        documents = [
            "Document one has multiple sentences. It discusses machine learning. ML is interesting.",
            "Document two covers different topics. It talks about deep learning. DL is powerful."
        ]
        summaries = summarizer.summarize_multiple(documents, num_sentences_per_doc=2)

        assert len(summaries) == 2
        assert all(isinstance(s, list) for s in summaries)

    def test_get_key_sentences(self):
        """Test getting sentences above threshold."""
        summarizer = ExtractiveSummarizer()
        text = """Machine learning is important. Deep learning uses networks.
        NLP processes language. AI transforms everything. Data is valuable."""
        key_sentences = summarizer.get_key_sentences(text, threshold=0.3)

        assert isinstance(key_sentences, list)
        assert all(isinstance(item, tuple) for item in key_sentences)
        assert all(len(item) == 2 for item in key_sentences)  # (sentence, score)
