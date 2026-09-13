"""Relevance Scoring System for Content Analysis.

Provides TF-IDF based relevance scoring for research content.
"""

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np


class RelevanceScorer:
    """Score content relevance based on keywords and TF-IDF."""

    def __init__(self, keywords: List[str] = None, stopwords: Set[str] = None):
        """Initialize the relevance scorer.

        Args:
            keywords: List of important keywords/topics to score against
            stopwords: Set of stopwords to exclude from analysis
        """
        self.keywords = [k.lower() for k in keywords] if keywords else []
        self.stopwords = stopwords or self._get_default_stopwords()
        self.idf_cache = {}
        self.corpus_size = 0

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
            text: Input text to tokenize

        Returns:
            List of lowercase tokens without stopwords
        """
        if not isinstance(text, str):
            return []

        # Extract words (3+ characters)
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Remove stopwords
        return [w for w in words if w not in self.stopwords]

    def compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency for tokens.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary mapping terms to their TF scores
        """
        if not tokens:
            return {}

        counter = Counter(tokens)
        max_freq = max(counter.values())

        # Normalized TF
        return {term: count / max_freq for term, count in counter.items()}

    def build_idf(self, documents: List[str]):
        """Build IDF cache from a corpus of documents.

        Args:
            documents: List of document texts
        """
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        # Count document frequency for each term
        df = defaultdict(int)
        for doc in documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                df[token] += 1

        # Compute IDF
        self.idf_cache = {
            term: math.log(self.corpus_size / (1 + freq))
            for term, freq in df.items()
        }

    def compute_tfidf(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF scores for text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping terms to TF-IDF scores
        """
        tokens = self.tokenize(text)
        tf = self.compute_tf(tokens)

        if not self.idf_cache:
            # If no IDF cache, just return TF
            return tf

        # Compute TF-IDF
        tfidf = {}
        for term, tf_val in tf.items():
            idf_val = self.idf_cache.get(term, math.log(self.corpus_size + 1))
            tfidf[term] = tf_val * idf_val

        return tfidf

    def keyword_density_score(self, text: str) -> float:
        """Calculate keyword density score.

        Args:
            text: Input text

        Returns:
            Keyword density score (0-1)
        """
        if not self.keywords:
            return 0.0

        tokens = self.tokenize(text)
        if not tokens:
            return 0.0

        keyword_count = sum(1 for token in tokens if token in self.keywords)
        return keyword_count / len(tokens)

    def keyword_coverage_score(self, text: str) -> float:
        """Calculate what fraction of keywords appear in text.

        Args:
            text: Input text

        Returns:
            Coverage score (0-1)
        """
        if not self.keywords:
            return 0.0

        tokens = set(self.tokenize(text))
        matched_keywords = sum(1 for kw in self.keywords if kw in tokens)
        return matched_keywords / len(self.keywords)

    def tfidf_keyword_score(self, text: str) -> float:
        """Calculate TF-IDF score for keywords in text.

        Args:
            text: Input text

        Returns:
            Average TF-IDF score for matched keywords
        """
        if not self.keywords:
            return 0.0

        tfidf = self.compute_tfidf(text)
        if not tfidf:
            return 0.0

        keyword_scores = [tfidf.get(kw, 0.0) for kw in self.keywords]
        matched_scores = [s for s in keyword_scores if s > 0]

        return sum(matched_scores) / len(self.keywords) if matched_scores else 0.0

    def calculate_relevance_score(
        self,
        text: str,
        weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive relevance score.

        Args:
            text: Input text
            weights: Optional weights for different scoring methods
                    Keys: 'density', 'coverage', 'tfidf'

        Returns:
            Dictionary with individual scores and combined score
        """
        # Default weights
        if weights is None:
            weights = {'density': 0.3, 'coverage': 0.4, 'tfidf': 0.3}

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate individual scores
        density = self.keyword_density_score(text)
        coverage = self.keyword_coverage_score(text)
        tfidf = self.tfidf_keyword_score(text)

        # Combined score
        combined = (
            weights.get('density', 0) * density +
            weights.get('coverage', 0) * coverage +
            weights.get('tfidf', 0) * tfidf
        )

        return {
            'density': density,
            'coverage': coverage,
            'tfidf': tfidf,
            'combined': combined
        }

    def score_documents(
        self,
        documents: List[str],
        weights: Dict[str, float] = None
    ) -> List[Dict[str, float]]:
        """Score multiple documents.

        Args:
            documents: List of document texts
            weights: Optional scoring weights

        Returns:
            List of score dictionaries for each document
        """
        # Build IDF from corpus
        self.build_idf(documents)

        # Score each document
        return [self.calculate_relevance_score(doc, weights) for doc in documents]

    def rank_documents(
        self,
        documents: List[Tuple[int, str]],
        weights: Dict[str, float] = None,
        top_k: int = None
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """Rank documents by relevance score.

        Args:
            documents: List of (id, text) tuples
            weights: Optional scoring weights
            top_k: Return only top K documents

        Returns:
            List of (id, combined_score, score_details) tuples, sorted by score
        """
        # Extract texts and build IDF
        texts = [doc[1] for doc in documents]
        self.build_idf(texts)

        # Score and attach IDs
        results = []
        for doc_id, text in documents:
            scores = self.calculate_relevance_score(text, weights)
            results.append((doc_id, scores['combined'], scores))

        # Sort by combined score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top K if specified
        if top_k:
            return results[:top_k]
        return results
