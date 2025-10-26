"""Integration tests for ContentAnalysisPipeline."""

import pytest
import json
import os
from pipeline import ContentAnalysisPipeline


class TestContentAnalysisPipeline:
    """Test ContentAnalysisPipeline integration."""

    def test_initialization(self, test_config, temp_db):
        """Test pipeline initialization."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)

        assert pipeline.config == test_config
        assert pipeline.db_path == temp_db
        assert pipeline.relevance_scorer is not None
        assert pipeline.summarizer is not None
        assert pipeline.duplicate_detector is not None
        assert pipeline.priority_classifier is not None

    def test_fetch_messages(self, test_config, temp_db):
        """Test fetching messages from database."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)
        messages = pipeline.fetch_messages(sender='assistant')

        assert len(messages) > 0
        assert all(isinstance(m[0], int) for m in messages)  # IDs
        assert all(isinstance(m[1], str) for m in messages)  # Content

    def test_fetch_messages_filtered(self, test_config, temp_db):
        """Test fetching filtered messages."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)

        claude_messages = pipeline.fetch_messages(platform='claude')
        chatgpt_messages = pipeline.fetch_messages(platform='chatgpt')

        assert len(claude_messages) > 0
        assert len(chatgpt_messages) > 0

    def test_fetch_messages_with_limit(self, test_config, temp_db):
        """Test message fetching with limit."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)
        messages = pipeline.fetch_messages(limit=2)

        assert len(messages) <= 2

    def test_analyze_relevance(self, test_config, sample_documents):
        """Test relevance analysis."""
        pipeline = ContentAnalysisPipeline(config=test_config)
        results = pipeline.analyze_relevance(sample_documents[:4])

        assert len(results) == 4
        for doc_id, scores in results.items():
            assert 'combined' in scores
            assert 0.0 <= scores['combined'] <= 1.0

    def test_generate_summaries(self, test_config, sample_documents):
        """Test summary generation."""
        pipeline = ContentAnalysisPipeline(config=test_config)
        summaries = pipeline.generate_summaries(sample_documents[:4], num_sentences=2)

        assert len(summaries) == 4
        for doc_id, summary in summaries.items():
            assert isinstance(summary, str)

    def test_detect_duplicates(self, test_config, sample_documents):
        """Test duplicate detection."""
        pipeline = ContentAnalysisPipeline(config=test_config)
        duplicates = pipeline.detect_duplicates(sample_documents)

        # Should find the duplicate pair (docs 0 and 5)
        assert len(duplicates) > 0
        assert all(isinstance(d[2], float) for d in duplicates)  # Similarity score

    def test_get_unique_documents(self, test_config, sample_documents):
        """Test filtering to unique documents."""
        pipeline = ContentAnalysisPipeline(config=test_config)
        unique_ids = pipeline.get_unique_documents(sample_documents)

        # Should be fewer than total (due to duplicate)
        assert len(unique_ids) < len(sample_documents)
        assert isinstance(unique_ids, list)

    def test_classify_priority(self, test_config, sample_documents):
        """Test priority classification."""
        pipeline = ContentAnalysisPipeline(config=test_config)
        priorities = pipeline.classify_priority(sample_documents[:4])

        assert len(priorities) == 4
        for doc_id, priority_info in priorities.items():
            assert 'level' in priority_info
            assert 'score' in priority_info
            assert 'reasons' in priority_info
            assert 0.0 <= priority_info['score'] <= 1.0

    def test_process_full_pipeline(self, test_config, temp_db):
        """Test complete pipeline processing."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)
        results = pipeline.process(limit=5)

        # Check structure
        assert 'metadata' in results
        assert 'documents' in results
        assert 'duplicates' in results
        assert 'statistics' in results

        # Check metadata
        assert 'timestamp' in results['metadata']
        assert 'total_documents' in results['metadata']
        assert 'processing_time' in results['metadata']

        # Check statistics
        assert 'relevance' in results['statistics']
        assert 'priority_distribution' in results['statistics']

        # Check documents
        for doc_id, doc_data in results['documents'].items():
            assert 'text' in doc_data
            assert 'relevance' in doc_data
            assert 'summary' in doc_data
            assert 'priority' in doc_data

    def test_process_with_skip_duplicates(self, test_config, temp_db):
        """Test pipeline with duplicate filtering."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)
        results = pipeline.process(skip_duplicates=True)

        # Should have fewer documents after filtering
        assert len(results['documents']) > 0

    def test_process_without_skip_duplicates(self, test_config, temp_db):
        """Test pipeline without duplicate filtering."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)
        results = pipeline.process(skip_duplicates=False)

        assert len(results['documents']) > 0

    def test_save_results(self, test_config, temp_db, tmp_path):
        """Test saving results to JSON."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)
        results = pipeline.process(limit=3)

        output_file = tmp_path / "test_results.json"
        pipeline.save_results(results, str(output_file))

        assert output_file.exists()

        # Verify JSON is valid
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        assert 'metadata' in loaded
        assert 'documents' in loaded

    def test_store_results_db(self, test_config, temp_db):
        """Test storing results back to database."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)
        results = pipeline.process(limit=3)

        # Store results
        pipeline.store_results_db(results, table_name='test_results')

        # Verify table was created and populated
        import sqlite3
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_results")
        count = cursor.fetchone()[0]
        conn.close()

        assert count > 0

    def test_compute_statistics(self, test_config, sample_documents):
        """Test statistics computation."""
        pipeline = ContentAnalysisPipeline(config=test_config)

        relevance_scores = {
            0: {'combined': 0.8},
            1: {'combined': 0.6},
            2: {'combined': 0.9}
        }

        priorities = {
            0: {'level': 'HIGH'},
            1: {'level': 'MEDIUM'},
            2: {'level': 'HIGH'}
        }

        stats = pipeline._compute_statistics(relevance_scores, priorities)

        assert 'relevance' in stats
        assert 'priority_distribution' in stats
        assert stats['relevance']['average'] > 0
        assert stats['priority_distribution']['HIGH'] == 2
        assert stats['priority_distribution']['MEDIUM'] == 1

    def test_pipeline_with_custom_config(self, temp_db):
        """Test pipeline with custom configuration."""
        custom_config = {
            'keywords': ['test', 'sample'],
            'summary_sentences': 1,
            'duplicate_threshold': 0.95,
        }

        pipeline = ContentAnalysisPipeline(config=custom_config, db_path=temp_db)
        results = pipeline.process(limit=3)

        assert results is not None
        assert len(results.get('documents', {})) > 0

    def test_pipeline_empty_database(self, test_config):
        """Test pipeline with empty/missing database."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path='nonexistent.db')
        messages = pipeline.fetch_messages()

        assert len(messages) == 0

    def test_pipeline_logging(self, test_config, temp_db):
        """Test that pipeline logs operations."""
        pipeline = ContentAnalysisPipeline(config=test_config, db_path=temp_db)

        assert pipeline.logger is not None
        assert pipeline.logger.name == 'ContentAnalysisPipeline'

        # Run pipeline to generate logs
        results = pipeline.process(limit=2)
        assert results is not None
