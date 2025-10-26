"""Tests for DuplicateDetector."""

import pytest
from pipeline.duplicate_detector import DuplicateDetector


class TestDuplicateDetector:
    """Test DuplicateDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = DuplicateDetector(similarity_threshold=0.8)
        assert detector.similarity_threshold == 0.8
        assert len(detector.stopwords) > 0

    def test_tokenize(self):
        """Test tokenization."""
        detector = DuplicateDetector()
        text = "Machine learning is fascinating and powerful."
        tokens = detector.tokenize(text)

        assert 'machine' in tokens
        assert 'learning' in tokens
        assert 'is' not in tokens  # Stopword

    def test_create_tf_vector(self):
        """Test TF vector creation."""
        detector = DuplicateDetector()
        tokens = ['machine', 'learning', 'machine', 'data']
        vec = detector.create_tf_vector(tokens)

        assert vec['machine'] == 0.5  # 2 out of 4
        assert vec['learning'] == 0.25  # 1 out of 4
        assert sum(vec.values()) == 1.0  # Normalized

    def test_cosine_similarity_identical(self):
        """Test cosine similarity for identical vectors."""
        detector = DuplicateDetector()
        vec = {'machine': 0.5, 'learning': 0.5}
        similarity = detector.cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 0.01  # Should be 1.0

    def test_cosine_similarity_different(self):
        """Test cosine similarity for different vectors."""
        detector = DuplicateDetector()
        vec1 = {'machine': 0.5, 'learning': 0.5}
        vec2 = {'dog': 0.5, 'cat': 0.5}
        similarity = detector.cosine_similarity(vec1, vec2)

        assert similarity == 0.0  # No common terms

    def test_text_similarity_duplicate(self):
        """Test similarity for near-duplicate texts."""
        detector = DuplicateDetector()
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "Machine learning is a subset of artificial intelligence."
        similarity = detector.text_similarity(text1, text2)

        assert similarity > 0.95  # Very high similarity

    def test_text_similarity_different(self):
        """Test similarity for different texts."""
        detector = DuplicateDetector()
        text1 = "Machine learning is important for AI."
        text2 = "Dogs and cats are popular pets."
        similarity = detector.text_similarity(text1, text2)

        assert similarity < 0.1  # Very low similarity

    def test_find_duplicates(self, sample_documents):
        """Test finding duplicate pairs."""
        detector = DuplicateDetector(similarity_threshold=0.9)
        duplicates = detector.find_duplicates(sample_documents)

        # Documents 0 and 5 are identical
        assert len(duplicates) > 0
        assert any(pair[0] == 0 and pair[1] == 5 for pair in duplicates) or \
               any(pair[0] == 5 and pair[1] == 0 for pair in duplicates)

    def test_find_near_duplicates(self, sample_documents):
        """Test finding near duplicates of a target."""
        detector = DuplicateDetector(similarity_threshold=0.7)
        # Find documents similar to document 0
        similar = detector.find_near_duplicates(sample_documents, target_id=0)

        # Should find document 5 (duplicate)
        similar_ids = [doc_id for doc_id, _ in similar]
        assert 5 in similar_ids

    def test_cluster_duplicates(self):
        """Test clustering of duplicates."""
        detector = DuplicateDetector(similarity_threshold=0.9)
        documents = [
            (0, "This is document A."),
            (1, "This is document A."),  # Duplicate of 0
            (2, "This is document B."),
            (3, "This is document B."),  # Duplicate of 2
            (4, "This is unique document C.")
        ]
        clusters = detector.cluster_duplicates(documents)

        # Should have 2 clusters (A-duplicates and B-duplicates)
        assert len(clusters) >= 2
        # Each cluster should have at least 2 members
        assert all(len(cluster) >= 2 for cluster in clusters)

    def test_get_unique_documents(self):
        """Test getting unique document IDs."""
        detector = DuplicateDetector(similarity_threshold=0.9)
        documents = [
            (0, "This is document A."),
            (1, "This is document A."),  # Duplicate
            (2, "This is document B."),
            (3, "This is completely different text here."),
        ]
        unique_ids = detector.get_unique_documents(documents)

        # Should have 3 unique documents (0 or 1, 2, 3)
        assert len(unique_ids) == 3
        assert 2 in unique_ids
        assert 3 in unique_ids

    def test_build_similarity_matrix(self):
        """Test building similarity matrix."""
        detector = DuplicateDetector()
        documents = [
            (0, "Machine learning is important."),
            (1, "Deep learning is powerful."),
            (2, "Natural language processing is useful.")
        ]
        matrix, doc_ids = detector.build_similarity_matrix(documents)

        assert matrix.shape == (3, 3)
        assert doc_ids == [0, 1, 2]
        # Diagonal should be 1.0 (self-similarity)
        assert abs(matrix[0][0] - 1.0) < 0.01
        assert abs(matrix[1][1] - 1.0) < 0.01
        # Symmetric
        assert abs(matrix[0][1] - matrix[1][0]) < 0.01
