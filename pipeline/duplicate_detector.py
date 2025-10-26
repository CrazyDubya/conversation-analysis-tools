"""Duplicate Detection using Cosine Similarity.

Identifies similar or duplicate content in document collections.
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
import numpy as np


class DuplicateDetector:
    """Detect duplicate or similar documents."""

    def __init__(self, stopwords: Set[str] = None, similarity_threshold: float = 0.8):
        """Initialize duplicate detector.

        Args:
            stopwords: Set of stopwords to exclude
            similarity_threshold: Threshold for considering documents similar (0-1)
        """
        self.stopwords = stopwords or self._get_default_stopwords()
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _get_default_stopwords() -> Set[str]:
        """Return default English stopwords."""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once'
        }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of lowercase tokens without stopwords
        """
        if not isinstance(text, str):
            return []

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in self.stopwords]

    def create_tf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Create term frequency vector.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary mapping terms to normalized frequencies
        """
        if not tokens:
            return {}

        counter = Counter(tokens)
        total = sum(counter.values())

        return {term: count / total for term, count in counter.items()}

    def cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float]
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector (term -> frequency)
            vec2: Second vector (term -> frequency)

        Returns:
            Cosine similarity (0-1)
        """
        if not vec1 or not vec2:
            return 0.0

        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())

        if not common_terms:
            return 0.0

        # Calculate dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        magnitude2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)

        vec1 = self.create_tf_vector(tokens1)
        vec2 = self.create_tf_vector(tokens2)

        return self.cosine_similarity(vec1, vec2)

    def find_duplicates(
        self,
        documents: List[Tuple[int, str]],
        threshold: float = None
    ) -> List[Tuple[int, int, float]]:
        """Find duplicate or similar document pairs.

        Args:
            documents: List of (id, text) tuples
            threshold: Similarity threshold (uses default if not provided)

        Returns:
            List of (id1, id2, similarity) tuples for similar pairs
        """
        if threshold is None:
            threshold = self.similarity_threshold

        duplicates = []
        n = len(documents)

        # Precompute TF vectors
        vectors = {}
        for doc_id, text in documents:
            tokens = self.tokenize(text)
            vectors[doc_id] = self.create_tf_vector(tokens)

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                id1, _ = documents[i]
                id2, _ = documents[j]

                similarity = self.cosine_similarity(vectors[id1], vectors[id2])

                if similarity >= threshold:
                    duplicates.append((id1, id2, similarity))

        # Sort by similarity (descending)
        duplicates.sort(key=lambda x: x[2], reverse=True)

        return duplicates

    def find_near_duplicates(
        self,
        documents: List[Tuple[int, str]],
        target_id: int
    ) -> List[Tuple[int, float]]:
        """Find documents similar to a target document.

        Args:
            documents: List of (id, text) tuples
            target_id: ID of target document to compare against

        Returns:
            List of (id, similarity) tuples, sorted by similarity
        """
        # Find target document
        target_text = None
        for doc_id, text in documents:
            if doc_id == target_id:
                target_text = text
                break

        if target_text is None:
            return []

        # Create target vector
        target_tokens = self.tokenize(target_text)
        target_vec = self.create_tf_vector(target_tokens)

        # Compare with all other documents
        similarities = []
        for doc_id, text in documents:
            if doc_id == target_id:
                continue

            tokens = self.tokenize(text)
            vec = self.create_tf_vector(tokens)

            similarity = self.cosine_similarity(target_vec, vec)

            if similarity >= self.similarity_threshold:
                similarities.append((doc_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def cluster_duplicates(
        self,
        documents: List[Tuple[int, str]],
        threshold: float = None
    ) -> List[Set[int]]:
        """Cluster documents into groups of duplicates.

        Uses single-linkage clustering based on similarity threshold.

        Args:
            documents: List of (id, text) tuples
            threshold: Similarity threshold (uses default if not provided)

        Returns:
            List of sets, each set contains IDs of similar documents
        """
        if threshold is None:
            threshold = self.similarity_threshold

        # Find all similar pairs
        pairs = self.find_duplicates(documents, threshold)

        # Build adjacency list
        graph = defaultdict(set)
        all_ids = set(doc_id for doc_id, _ in documents)

        for id1, id2, _ in pairs:
            graph[id1].add(id2)
            graph[id2].add(id1)

        # Find connected components (clusters)
        visited = set()
        clusters = []

        def dfs(node, cluster):
            visited.add(node)
            cluster.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for doc_id in all_ids:
            if doc_id not in visited:
                cluster = set()
                dfs(doc_id, cluster)
                if len(cluster) > 1:  # Only include actual clusters
                    clusters.append(cluster)

        return clusters

    def get_unique_documents(
        self,
        documents: List[Tuple[int, str]],
        threshold: float = None
    ) -> List[int]:
        """Get list of unique document IDs (one per cluster).

        For each cluster of duplicates, keeps the first document.

        Args:
            documents: List of (id, text) tuples
            threshold: Similarity threshold (uses default if not provided)

        Returns:
            List of document IDs representing unique documents
        """
        clusters = self.cluster_duplicates(documents, threshold)

        # Get all document IDs
        all_ids = set(doc_id for doc_id, _ in documents)

        # IDs in clusters (mark duplicates)
        clustered_ids = set()
        for cluster in clusters:
            clustered_ids.update(cluster)

        # Keep one ID from each cluster (lowest ID)
        unique_ids = [min(cluster) for cluster in clusters]

        # Add standalone documents (not in any cluster)
        standalone = all_ids - clustered_ids
        unique_ids.extend(standalone)

        return sorted(unique_ids)

    def build_similarity_matrix(
        self,
        documents: List[Tuple[int, str]]
    ) -> Tuple[np.ndarray, List[int]]:
        """Build full similarity matrix for documents.

        Args:
            documents: List of (id, text) tuples

        Returns:
            Tuple of (similarity_matrix, document_ids)
        """
        n = len(documents)
        matrix = np.zeros((n, n))
        doc_ids = [doc_id for doc_id, _ in documents]

        # Precompute TF vectors
        vectors = {}
        for doc_id, text in documents:
            tokens = self.tokenize(text)
            vectors[doc_id] = self.create_tf_vector(tokens)

        # Compute similarities
        for i in range(n):
            for j in range(i + 1, n):
                id1 = doc_ids[i]
                id2 = doc_ids[j]

                similarity = self.cosine_similarity(vectors[id1], vectors[id2])
                matrix[i][j] = similarity
                matrix[j][i] = similarity

        # Diagonal is 1.0 (self-similarity)
        np.fill_diagonal(matrix, 1.0)

        return matrix, doc_ids
