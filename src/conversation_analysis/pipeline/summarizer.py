"""Extractive Summarization using TextRank algorithm.

Provides sentence extraction based on importance scoring.
"""

import re
import math
from collections import defaultdict
from typing import List, Tuple, Set
import numpy as np


class ExtractiveSummarizer:
    """Extract key sentences from text using TextRank."""

    def __init__(self, stopwords: Set[str] = None, damping: float = 0.85):
        """Initialize the summarizer.

        Args:
            stopwords: Set of stopwords to exclude
            damping: Damping factor for PageRank (default: 0.85)
        """
        self.stopwords = stopwords or self._get_default_stopwords()
        self.damping = damping

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

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not isinstance(text, str):
            return []

        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)

        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]

        # Filter out very short sentences
        return [s for s in sentences if len(s.split()) >= 3]

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

    def sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences.

        Uses word overlap (cosine similarity of word vectors).

        Args:
            sent1: First sentence
            sent2: Second sentence

        Returns:
            Similarity score (0-1)
        """
        tokens1 = set(self.tokenize(sent1))
        tokens2 = set(self.tokenize(sent2))

        if not tokens1 or not tokens2:
            return 0.0

        # Cosine similarity using word overlap
        intersection = len(tokens1 & tokens2)
        denominator = math.sqrt(len(tokens1) * len(tokens2))

        return intersection / denominator if denominator > 0 else 0.0

    def build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Build sentence similarity matrix.

        Args:
            sentences: List of sentences

        Returns:
            NxN similarity matrix
        """
        n = len(sentences)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.sentence_similarity(sentences[i], sentences[j])
                matrix[i][j] = sim
                matrix[j][i] = sim

        return matrix

    def textrank(self, similarity_matrix: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply TextRank algorithm (PageRank for sentences).

        Args:
            similarity_matrix: Sentence similarity matrix
            max_iter: Maximum iterations

        Returns:
            Array of sentence scores
        """
        n = similarity_matrix.shape[0]
        if n == 0:
            return np.array([])

        # Initialize scores
        scores = np.ones(n) / n

        # Normalize similarity matrix (row-wise)
        row_sums = similarity_matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        normalized_matrix = similarity_matrix / row_sums[:, np.newaxis]

        # Iterate
        for _ in range(max_iter):
            prev_scores = scores.copy()

            # PageRank formula
            scores = (
                (1 - self.damping) / n +
                self.damping * normalized_matrix.T @ prev_scores
            )

            # Check convergence
            if np.allclose(scores, prev_scores, atol=1e-6):
                break

        return scores

    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        return_scores: bool = False
    ) -> Tuple[List[str], List[float]] or List[str]:
        """Generate extractive summary.

        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            return_scores: Whether to return sentence scores

        Returns:
            List of summary sentences (and optionally their scores)
        """
        # Split into sentences
        sentences = self.split_sentences(text)

        if len(sentences) == 0:
            return ([], []) if return_scores else []

        if len(sentences) <= num_sentences:
            # Return all sentences if already short
            scores = [1.0] * len(sentences)
            return (sentences, scores) if return_scores else sentences

        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(sentences)

        # Apply TextRank
        scores = self.textrank(similarity_matrix)

        # Rank sentences
        ranked_indices = np.argsort(scores)[::-1]

        # Select top sentences
        top_indices = sorted(ranked_indices[:num_sentences])
        summary_sentences = [sentences[i] for i in top_indices]
        summary_scores = [scores[i] for i in top_indices]

        if return_scores:
            return summary_sentences, summary_scores
        return summary_sentences

    def summarize_multiple(
        self,
        documents: List[str],
        num_sentences_per_doc: int = 3
    ) -> List[List[str]]:
        """Summarize multiple documents.

        Args:
            documents: List of document texts
            num_sentences_per_doc: Sentences to extract per document

        Returns:
            List of summaries (each summary is a list of sentences)
        """
        return [
            self.summarize(doc, num_sentences=num_sentences_per_doc)
            for doc in documents
        ]

    def summarize_to_text(
        self,
        text: str,
        num_sentences: int = 3,
        separator: str = ' '
    ) -> str:
        """Generate summary as single text string.

        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            separator: String to join sentences (default: space)

        Returns:
            Summary text
        """
        summary_sentences = self.summarize(text, num_sentences)
        return separator.join(summary_sentences)

    def get_key_sentences(
        self,
        text: str,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Get sentences above importance threshold.

        Args:
            text: Input text
            threshold: Minimum score threshold (0-1)

        Returns:
            List of (sentence, score) tuples
        """
        sentences = self.split_sentences(text)

        if len(sentences) == 0:
            return []

        if len(sentences) == 1:
            return [(sentences[0], 1.0)]

        # Build similarity matrix and get scores
        similarity_matrix = self.build_similarity_matrix(sentences)
        scores = self.textrank(similarity_matrix)

        # Normalize scores to 0-1
        if scores.max() > 0:
            scores = scores / scores.max()

        # Filter by threshold
        results = [
            (sent, score)
            for sent, score in zip(sentences, scores)
            if score >= threshold
        ]

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results
