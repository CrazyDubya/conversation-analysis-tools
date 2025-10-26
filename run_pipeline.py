#!/usr/bin/env python3
"""Example script to run the content analysis pipeline."""

import yaml
import argparse
import logging
from pathlib import Path
from pipeline import ContentAnalysisPipeline


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/pipeline.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run content analysis pipeline')
    parser.add_argument(
        '--config',
        default='config/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--db',
        default='conversations.db',
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--platform',
        choices=['claude', 'chatgpt', 'all'],
        default='all',
        help='Platform to analyze'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of messages to process'
    )
    parser.add_argument(
        '--output',
        default='output/pipeline_results.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--no-save-db',
        action='store_true',
        help='Do not save results to database'
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Setup logging
    log_level = config.get('output', {}).get('log_level', 'INFO')
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    # Initialize pipeline
    logger.info("Initializing Content Analysis Pipeline...")
    pipeline = ContentAnalysisPipeline(config=config, db_path=args.db)

    # Determine platform
    platform = None if args.platform == 'all' else args.platform

    # Run pipeline
    logger.info(f"Processing messages from {platform or 'all platforms'}...")
    results = pipeline.process(
        platform=platform,
        limit=args.limit,
        skip_duplicates=config.get('processing', {}).get('skip_duplicates', True)
    )

    if not results:
        logger.warning("No results generated")
        return

    # Display statistics
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)
    print(f"Total documents processed: {results['metadata']['total_documents']}")
    print(f"Duplicate pairs found: {results['metadata']['duplicate_pairs']}")
    print(f"Processing time: {results['metadata']['processing_time']:.2f}s")
    print("\nStatistics:")
    print(f"  Average relevance: {results['statistics']['relevance']['average']:.3f}")
    print(f"  Relevance range: [{results['statistics']['relevance']['min']:.3f}, {results['statistics']['relevance']['max']:.3f}]")
    print("\nPriority distribution:")
    for level, count in results['statistics']['priority_distribution'].items():
        print(f"  {level}: {count}")

    # Save results
    print(f"\nSaving results to {args.output}...")
    pipeline.save_results(results, args.output)

    # Save to database
    if not args.no_save_db:
        table_name = config.get('database', {}).get('results_table', 'analysis_results')
        print(f"Storing results in database table '{table_name}'...")
        pipeline.store_results_db(results, table_name)

    print("\n✓ Pipeline execution complete!")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
