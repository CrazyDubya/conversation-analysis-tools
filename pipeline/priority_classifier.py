"""Priority Classification System.

Assigns priority levels to documents based on multiple criteria.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class PriorityLevel(Enum):
    """Priority levels for documents."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NONE = 0


@dataclass
class PriorityScore:
    """Priority scoring result."""
    level: PriorityLevel
    score: float
    reasons: List[str]
    metadata: Dict


class PriorityClassifier:
    """Classify document priority based on multiple factors."""

    def __init__(self, config: Dict = None):
        """Initialize priority classifier.

        Args:
            config: Configuration dictionary with thresholds and weights
        """
        self.config = config or self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict:
        """Return default configuration."""
        return {
            # Thresholds for different metrics
            'relevance_thresholds': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            },
            'length_thresholds': {
                'min_words': 50,      # Minimum for consideration
                'substantial': 200,    # Substantial content
                'comprehensive': 500   # Comprehensive content
            },
            'keyword_thresholds': {
                'critical_keywords': [],  # Keywords indicating critical content
                'high_keywords': [],       # Keywords indicating high priority
                'urgent_patterns': []      # Regex patterns for urgency
            },
            # Weights for combined scoring
            'weights': {
                'relevance': 0.4,
                'length': 0.2,
                'keyword_match': 0.3,
                'recency': 0.1
            },
            # Score thresholds for priority levels
            'priority_thresholds': {
                'critical': 0.85,
                'high': 0.65,
                'medium': 0.45,
                'low': 0.25
            }
        }

    def calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length.

        Longer, more comprehensive texts score higher.

        Args:
            text: Input text

        Returns:
            Length score (0-1)
        """
        if not isinstance(text, str):
            return 0.0

        word_count = len(text.split())
        thresholds = self.config['length_thresholds']

        if word_count < thresholds['min_words']:
            return 0.1
        elif word_count < thresholds['substantial']:
            return 0.5
        elif word_count < thresholds['comprehensive']:
            return 0.8
        else:
            return 1.0

    def calculate_keyword_score(
        self,
        text: str,
        critical_keywords: List[str] = None,
        high_keywords: List[str] = None
    ) -> Tuple[float, List[str]]:
        """Calculate score based on keyword presence.

        Args:
            text: Input text
            critical_keywords: List of critical keywords
            high_keywords: List of high-priority keywords

        Returns:
            Tuple of (score, matched_keywords)
        """
        if not isinstance(text, str):
            return 0.0, []

        text_lower = text.lower()
        matched = []
        score = 0.0

        # Check critical keywords
        critical_kw = critical_keywords or self.config['keyword_thresholds']['critical_keywords']
        for kw in critical_kw:
            if kw.lower() in text_lower:
                score = max(score, 1.0)
                matched.append(f"critical: {kw}")

        # Check high-priority keywords
        high_kw = high_keywords or self.config['keyword_thresholds']['high_keywords']
        for kw in high_kw:
            if kw.lower() in text_lower:
                score = max(score, 0.7)
                matched.append(f"high: {kw}")

        return score, matched

    def calculate_combined_score(
        self,
        relevance_score: float,
        length_score: float,
        keyword_score: float,
        recency_score: float = 0.5,
        weights: Dict[str, float] = None
    ) -> float:
        """Calculate combined priority score.

        Args:
            relevance_score: Relevance score (0-1)
            length_score: Length score (0-1)
            keyword_score: Keyword match score (0-1)
            recency_score: Recency score (0-1), default 0.5
            weights: Custom weights (optional)

        Returns:
            Combined score (0-1)
        """
        w = weights or self.config['weights']

        # Normalize weights
        total_weight = sum(w.values())
        if total_weight > 0:
            w = {k: v / total_weight for k, v in w.items()}

        combined = (
            w.get('relevance', 0) * relevance_score +
            w.get('length', 0) * length_score +
            w.get('keyword_match', 0) * keyword_score +
            w.get('recency', 0) * recency_score
        )

        return combined

    def score_to_priority(self, score: float) -> PriorityLevel:
        """Convert numeric score to priority level.

        Args:
            score: Combined priority score (0-1)

        Returns:
            PriorityLevel enum
        """
        thresholds = self.config['priority_thresholds']

        if score >= thresholds['critical']:
            return PriorityLevel.CRITICAL
        elif score >= thresholds['high']:
            return PriorityLevel.HIGH
        elif score >= thresholds['medium']:
            return PriorityLevel.MEDIUM
        elif score >= thresholds['low']:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.NONE

    def classify(
        self,
        text: str,
        relevance_score: float = None,
        metadata: Dict = None
    ) -> PriorityScore:
        """Classify document priority.

        Args:
            text: Document text
            relevance_score: Optional pre-computed relevance score
            metadata: Optional metadata (for recency, etc.)

        Returns:
            PriorityScore object
        """
        metadata = metadata or {}
        reasons = []

        # Calculate component scores
        length_score = self.calculate_length_score(text)
        keyword_score, matched_keywords = self.calculate_keyword_score(text)

        # Use provided relevance or default
        if relevance_score is None:
            relevance_score = 0.5  # Neutral if not provided

        # Recency score (can be enhanced with timestamps)
        recency_score = metadata.get('recency_score', 0.5)

        # Build reasons list
        if length_score >= 0.8:
            reasons.append("Comprehensive content (500+ words)")
        elif length_score >= 0.5:
            reasons.append("Substantial content (200+ words)")

        if keyword_score >= 0.7:
            reasons.append(f"Contains priority keywords: {', '.join(matched_keywords)}")

        if relevance_score >= 0.7:
            reasons.append(f"High relevance score ({relevance_score:.2f})")

        # Calculate combined score
        combined_score = self.calculate_combined_score(
            relevance_score,
            length_score,
            keyword_score,
            recency_score
        )

        # Determine priority level
        priority_level = self.score_to_priority(combined_score)

        # Add priority level reason
        reasons.insert(0, f"Overall priority score: {combined_score:.2f}")

        return PriorityScore(
            level=priority_level,
            score=combined_score,
            reasons=reasons,
            metadata={
                'relevance_score': relevance_score,
                'length_score': length_score,
                'keyword_score': keyword_score,
                'recency_score': recency_score,
                'matched_keywords': matched_keywords
            }
        )

    def classify_batch(
        self,
        documents: List[Tuple[int, str, float]],
        metadata_list: List[Dict] = None
    ) -> List[Tuple[int, PriorityScore]]:
        """Classify multiple documents.

        Args:
            documents: List of (id, text, relevance_score) tuples
            metadata_list: Optional list of metadata dicts

        Returns:
            List of (id, PriorityScore) tuples
        """
        if metadata_list is None:
            metadata_list = [{}] * len(documents)

        results = []
        for (doc_id, text, relevance), metadata in zip(documents, metadata_list):
            priority = self.classify(text, relevance, metadata)
            results.append((doc_id, priority))

        return results

    def rank_by_priority(
        self,
        documents: List[Tuple[int, str, float]],
        metadata_list: List[Dict] = None,
        filter_level: PriorityLevel = None
    ) -> List[Tuple[int, PriorityScore]]:
        """Rank documents by priority.

        Args:
            documents: List of (id, text, relevance_score) tuples
            metadata_list: Optional metadata list
            filter_level: Optional minimum priority level filter

        Returns:
            Sorted list of (id, PriorityScore) tuples
        """
        # Classify all documents
        results = self.classify_batch(documents, metadata_list)

        # Filter by priority level if specified
        if filter_level:
            results = [
                (doc_id, priority)
                for doc_id, priority in results
                if priority.level.value >= filter_level.value
            ]

        # Sort by score (descending)
        results.sort(key=lambda x: x[1].score, reverse=True)

        return results

    def get_priority_distribution(
        self,
        documents: List[Tuple[int, str, float]],
        metadata_list: List[Dict] = None
    ) -> Dict[PriorityLevel, int]:
        """Get distribution of priorities across documents.

        Args:
            documents: List of (id, text, relevance_score) tuples
            metadata_list: Optional metadata list

        Returns:
            Dictionary mapping priority levels to counts
        """
        results = self.classify_batch(documents, metadata_list)

        distribution = {level: 0 for level in PriorityLevel}
        for _, priority in results:
            distribution[priority.level] += 1

        return distribution
