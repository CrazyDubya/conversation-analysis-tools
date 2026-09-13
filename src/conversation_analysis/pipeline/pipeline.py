"""Main Content Analysis Pipeline.

Orchestrates all analysis modules for end-to-end processing.
"""

import sqlite3
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json

from .relevance_scorer import RelevanceScorer
from .summarizer import ExtractiveSummarizer
from .duplicate_detector import DuplicateDetector
from .priority_classifier import PriorityClassifier, PriorityLevel


class ContentAnalysisPipeline:
    """Main pipeline for content analysis."""

    def __init__(self, config: Dict = None, db_path: str = None):
        """Initialize the pipeline.

        Args:
            config: Configuration dictionary
            db_path: Path to SQLite database
        """
        self.config = config or {}
        self.db_path = db_path
        self.logger = self._setup_logger()

        # Initialize modules
        self.relevance_scorer = RelevanceScorer(
            keywords=self.config.get('keywords', []),
            stopwords=None
        )

        self.summarizer = ExtractiveSummarizer(
            stopwords=None,
            damping=self.config.get('summarizer', {}).get('damping', 0.85)
        )

        self.duplicate_detector = DuplicateDetector(
            stopwords=None,
            similarity_threshold=self.config.get('duplicate_threshold', 0.8)
        )

        self.priority_classifier = PriorityClassifier(
            config=self.config.get('priority', {})
        )

        self.logger.info("Content Analysis Pipeline initialized")

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for pipeline."""
        logger = logging.getLogger('ContentAnalysisPipeline')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def fetch_messages(
        self,
        platform: str = None,
        sender: str = 'assistant',
        limit: int = None
    ) -> List[Tuple[int, str]]:
        """Fetch messages from database.

        Args:
            platform: Filter by platform (claude/chatgpt)
            sender: Filter by sender (assistant/user)
            limit: Maximum number of messages

        Returns:
            List of (message_id, content) tuples
        """
        if not self.db_path:
            self.logger.error("No database path configured")
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
                SELECT m.id, m.content
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.sender = ?
            """
            params = [sender]

            if platform:
                query += " AND c.platform = ?"
                params.append(platform)

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            messages = cursor.fetchall()
            conn.close()

            self.logger.info(f"Fetched {len(messages)} messages from database")
            return messages

        except Exception as e:
            self.logger.error(f"Error fetching messages: {e}")
            return []

    def analyze_relevance(
        self,
        documents: List[Tuple[int, str]],
        weights: Dict[str, float] = None
    ) -> Dict[int, Dict[str, float]]:
        """Analyze relevance scores for documents.

        Args:
            documents: List of (id, text) tuples
            weights: Optional scoring weights

        Returns:
            Dictionary mapping document IDs to score dictionaries
        """
        self.logger.info(f"Analyzing relevance for {len(documents)} documents")

        ranked = self.relevance_scorer.rank_documents(documents, weights)

        results = {}
        for doc_id, combined_score, scores in ranked:
            results[doc_id] = scores

        self.logger.info("Relevance analysis complete")
        return results

    def generate_summaries(
        self,
        documents: List[Tuple[int, str]],
        num_sentences: int = 3
    ) -> Dict[int, str]:
        """Generate summaries for documents.

        Args:
            documents: List of (id, text) tuples
            num_sentences: Number of sentences to extract

        Returns:
            Dictionary mapping document IDs to summaries
        """
        self.logger.info(f"Generating summaries for {len(documents)} documents")

        summaries = {}
        for doc_id, text in documents:
            summary = self.summarizer.summarize_to_text(text, num_sentences)
            summaries[doc_id] = summary

        self.logger.info("Summary generation complete")
        return summaries

    def detect_duplicates(
        self,
        documents: List[Tuple[int, str]],
        threshold: float = None
    ) -> List[Tuple[int, int, float]]:
        """Detect duplicate documents.

        Args:
            documents: List of (id, text) tuples
            threshold: Similarity threshold

        Returns:
            List of (id1, id2, similarity) tuples
        """
        self.logger.info(f"Detecting duplicates in {len(documents)} documents")

        duplicates = self.duplicate_detector.find_duplicates(documents, threshold)

        self.logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates

    def get_unique_documents(
        self,
        documents: List[Tuple[int, str]],
        threshold: float = None
    ) -> List[int]:
        """Get unique document IDs (filtering duplicates).

        Args:
            documents: List of (id, text) tuples
            threshold: Similarity threshold

        Returns:
            List of unique document IDs
        """
        unique_ids = self.duplicate_detector.get_unique_documents(documents, threshold)
        self.logger.info(f"Filtered to {len(unique_ids)} unique documents")
        return unique_ids

    def classify_priority(
        self,
        documents: List[Tuple[int, str]],
        relevance_scores: Dict[int, Dict[str, float]] = None
    ) -> Dict[int, Dict]:
        """Classify document priorities.

        Args:
            documents: List of (id, text) tuples
            relevance_scores: Optional pre-computed relevance scores

        Returns:
            Dictionary mapping document IDs to priority info
        """
        self.logger.info(f"Classifying priority for {len(documents)} documents")

        # Prepare data for batch classification
        batch_data = []
        for doc_id, text in documents:
            relevance = 0.5  # Default
            if relevance_scores and doc_id in relevance_scores:
                relevance = relevance_scores[doc_id].get('combined', 0.5)
            batch_data.append((doc_id, text, relevance))

        # Classify
        results = self.priority_classifier.classify_batch(batch_data)

        # Format results
        priorities = {}
        for doc_id, priority_score in results:
            priorities[doc_id] = {
                'level': priority_score.level.name,
                'score': priority_score.score,
                'reasons': priority_score.reasons,
                'metadata': priority_score.metadata
            }

        self.logger.info("Priority classification complete")
        return priorities

    def process(
        self,
        platform: str = None,
        limit: int = None,
        skip_duplicates: bool = True
    ) -> Dict:
        """Run complete analysis pipeline.

        Args:
            platform: Filter by platform
            limit: Maximum number of messages
            skip_duplicates: Whether to filter duplicates

        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting content analysis pipeline")
        start_time = datetime.now()

        # 1. Fetch messages
        documents = self.fetch_messages(platform, limit=limit)
        if not documents:
            self.logger.warning("No documents to process")
            return {}

        self.logger.info(f"Processing {len(documents)} documents")

        # 2. Analyze relevance
        relevance_scores = self.analyze_relevance(documents)

        # 3. Detect duplicates
        duplicate_pairs = self.detect_duplicates(documents)

        # 4. Filter to unique documents if requested
        if skip_duplicates:
            unique_ids = self.get_unique_documents(documents)
            documents = [(doc_id, text) for doc_id, text in documents if doc_id in unique_ids]
            self.logger.info(f"Filtered to {len(documents)} unique documents")

        # 5. Generate summaries
        summaries = self.generate_summaries(
            documents,
            num_sentences=self.config.get('summary_sentences', 3)
        )

        # 6. Classify priorities
        priorities = self.classify_priority(documents, relevance_scores)

        # 7. Compile results
        results = {
            'metadata': {
                'timestamp': start_time.isoformat(),
                'platform': platform,
                'total_documents': len(documents),
                'duplicate_pairs': len(duplicate_pairs),
                'processing_time': (datetime.now() - start_time).total_seconds()
            },
            'documents': {},
            'duplicates': duplicate_pairs,
            'statistics': self._compute_statistics(relevance_scores, priorities)
        }

        # Organize by document
        for doc_id, text in documents:
            results['documents'][doc_id] = {
                'text': text[:500] + '...' if len(text) > 500 else text,  # Truncate
                'relevance': relevance_scores.get(doc_id, {}),
                'summary': summaries.get(doc_id, ''),
                'priority': priorities.get(doc_id, {})
            }

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.logger.info(f"Pipeline complete in {duration:.2f} seconds")

        return results

    def _compute_statistics(
        self,
        relevance_scores: Dict[int, Dict[str, float]],
        priorities: Dict[int, Dict]
    ) -> Dict:
        """Compute summary statistics."""
        stats = {
            'relevance': {
                'average': 0.0,
                'min': 1.0,
                'max': 0.0
            },
            'priority_distribution': {
                level: 0 for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NONE']
            }
        }

        # Relevance stats
        if relevance_scores:
            combined_scores = [s.get('combined', 0) for s in relevance_scores.values()]
            stats['relevance']['average'] = sum(combined_scores) / len(combined_scores)
            stats['relevance']['min'] = min(combined_scores)
            stats['relevance']['max'] = max(combined_scores)

        # Priority distribution
        for priority_info in priorities.values():
            level = priority_info.get('level', 'NONE')
            stats['priority_distribution'][level] += 1

        return stats

    def save_results(
        self,
        results: Dict,
        output_path: str = 'pipeline_results.json'
    ):
        """Save results to JSON file.

        Args:
            results: Results dictionary from process()
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def store_results_db(self, results: Dict, table_name: str = 'analysis_results'):
        """Store results back to database.

        Args:
            results: Results dictionary from process()
            table_name: Name of table to store results
        """
        if not self.db_path:
            self.logger.error("No database path configured")
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create results table if not exists
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    message_id INTEGER PRIMARY KEY,
                    relevance_score REAL,
                    summary TEXT,
                    priority_level TEXT,
                    priority_score REAL,
                    analyzed_at TEXT,
                    metadata TEXT
                )
            """)

            # Insert results
            for doc_id, doc_data in results['documents'].items():
                relevance = doc_data.get('relevance', {}).get('combined', 0.0)
                summary = doc_data.get('summary', '')
                priority_info = doc_data.get('priority', {})
                priority_level = priority_info.get('level', 'NONE')
                priority_score = priority_info.get('score', 0.0)

                cursor.execute(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (message_id, relevance_score, summary, priority_level, priority_score, analyzed_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    relevance,
                    summary,
                    priority_level,
                    priority_score,
                    results['metadata']['timestamp'],
                    json.dumps(priority_info.get('metadata', {}))
                ))

            conn.commit()
            conn.close()
            self.logger.info(f"Results stored in database table '{table_name}'")

        except Exception as e:
            self.logger.error(f"Error storing results in database: {e}")
