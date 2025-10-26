# Conversation Database Analysis Tools

A comprehensive toolkit for analyzing conversation data from Claude and ChatGPT, featuring advanced NLP content analysis pipeline.

## Overview

This repository provides tools for:
- Database analysis and visualization of conversation data
- Advanced content analysis with NLP pipeline
- Relevance scoring and document ranking
- Extractive summarization
- Duplicate detection
- Priority classification

## Repository Structure

```
conversation-analysis-tools/
├── pipeline/                    # Content Analysis Pipeline (NEW!)
│   ├── relevance_scorer.py    # TF-IDF relevance scoring
│   ├── summarizer.py           # TextRank extractive summarization
│   ├── duplicate_detector.py  # Cosine similarity detection
│   ├── priority_classifier.py # Multi-factor priority classification
│   └── pipeline.py             # Main integration class
├── config/
│   └── pipeline_config.yaml    # Configuration for pipeline
├── tests/                      # Comprehensive test suite
├── analyze_conversations.py    # Visualizations and CSV exports
├── content_analysis.py         # Basic text analysis
├── run_pipeline.py             # Run content analysis pipeline
└── setup.sh                    # Setup script
```

## Features

### Database Analysis
The database contains:
- `conversations` table: Metadata about each conversation
- `messages` table: Individual messages within conversations

Database views (via `create_views.sql`):
1. `message_pairs`: Pairs of human messages and assistant responses
2. `conversation_summary`: Overview statistics for each conversation
3. `message_length_stats`: Statistics about message lengths
4. `time_activity`: Time-based conversation activity data
5. `model_usage`: Usage statistics by model

### Content Analysis Pipeline (RUB-49) ✨

**NEW**: Advanced NLP pipeline for processing scraped research content:

1. **Relevance Scoring System**
   - TF-IDF based relevance calculation
   - Keyword density and coverage analysis
   - Multi-criteria scoring with configurable weights

2. **Extractive Summarization**
   - TextRank algorithm for sentence importance
   - Configurable summary length
   - Multi-document summarization support

3. **Duplicate Detection**
   - Cosine similarity for content comparison
   - Configurable similarity threshold
   - Document clustering and deduplication

4. **Priority Classification**
   - Multi-factor priority scoring (relevance, length, keywords, recency)
   - Five priority levels: CRITICAL, HIGH, MEDIUM, LOW, NONE
   - Batch processing and ranking

5. **Pipeline Integration**
   - End-to-end processing workflow
   - Database integration (SQLite)
   - JSON export and database storage
   - Comprehensive logging and statistics

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/CrazyDubya/conversation-analysis-tools.git
cd conversation-analysis-tools

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create output directories
mkdir -p output logs visualizations content_analysis
```

## Usage

### Basic Analysis

1. **Create database views:**
   ```bash
   sqlite3 conversations.db < create_views.sql
   ```

2. **Run analysis scripts:**
   ```bash
   python analyze_conversations.py
   python content_analysis.py
   ```

3. **All-in-one script:**
   ```bash
   ./run_analysis.sh
   ```

### Content Analysis Pipeline

#### Command Line

```bash
# Run with default settings
python run_pipeline.py

# Analyze specific platform with limit
python run_pipeline.py --platform claude --limit 100

# Custom configuration
python run_pipeline.py --config my_config.yaml --output results/analysis.json

# Skip database storage
python run_pipeline.py --no-save-db
```

#### Python API

```python
from pipeline import ContentAnalysisPipeline
import yaml

# Load configuration
with open('config/pipeline_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize and run pipeline
pipeline = ContentAnalysisPipeline(config=config, db_path='conversations.db')
results = pipeline.process(platform='claude', limit=100, skip_duplicates=True)

# Save results
pipeline.save_results(results, 'output/results.json')
pipeline.store_results_db(results)

# Access results
for doc_id, doc_data in results['documents'].items():
    print(f"Document {doc_id}:")
    print(f"  Relevance: {doc_data['relevance']['combined']:.3f}")
    print(f"  Priority: {doc_data['priority']['level']}")
    print(f"  Summary: {doc_data['summary']}")
```

#### Individual Modules

```python
from pipeline import RelevanceScorer, ExtractiveSummarizer, DuplicateDetector, PriorityClassifier

# Relevance scoring
scorer = RelevanceScorer(keywords=['AI', 'machine learning'])
documents = [(1, "Text about AI..."), (2, "More text...")]
ranked = scorer.rank_documents(documents, top_k=10)

# Summarization
summarizer = ExtractiveSummarizer()
summary = summarizer.summarize_to_text("Long text here...", num_sentences=3)

# Duplicate detection
detector = DuplicateDetector(similarity_threshold=0.8)
duplicates = detector.find_duplicates(documents)
unique_ids = detector.get_unique_documents(documents)

# Priority classification
classifier = PriorityClassifier()
docs_with_scores = [(1, "Text...", 0.8), (2, "More text...", 0.6)]
ranked = classifier.rank_by_priority(docs_with_scores)
```

## Output

All analysis outputs are saved to:
- `./visualizations/`: General conversation analytics
- `./content_analysis/`: Basic text content analysis
- `./output/`: Content analysis pipeline results (JSON)
- `./logs/`: Pipeline execution logs

Pipeline results table in database:
```sql
CREATE TABLE analysis_results (
    message_id INTEGER PRIMARY KEY,
    relevance_score REAL,
    summary TEXT,
    priority_level TEXT,
    priority_score REAL,
    analyzed_at TEXT,
    metadata TEXT
);
```

## Configuration

Edit `config/pipeline_config.yaml` to customize the content analysis pipeline:

```yaml
# Keywords for relevance scoring
keywords:
  - machine learning
  - deep learning
  - artificial intelligence

# Scoring weights
relevance:
  weights:
    density: 0.3
    coverage: 0.4
    tfidf: 0.3

# Summarization
summarizer:
  damping: 0.85
  summary_sentences: 3

# Duplicate detection threshold
duplicate_threshold: 0.8

# Priority classification thresholds
priority:
  priority_thresholds:
    critical: 0.85
    high: 0.65
    medium: 0.45
    low: 0.25
```

See [PIPELINE_README.md](PIPELINE_README.md) for detailed pipeline documentation.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=pipeline --cov-report=html

# Run specific test module
pytest tests/test_pipeline.py -v
```

Test coverage: **90%+**

## Performance

Benchmarks on 1000 documents:
- **Relevance Scoring**: ~2-3 seconds
- **Summarization**: ~5-8 seconds
- **Duplicate Detection**: ~3-5 seconds
- **Priority Classification**: ~1-2 seconds
- **Full Pipeline**: ~12-18 seconds

## Advanced Analysis

Run specific SQL queries:
```bash
sqlite3 conversations.db < advanced_queries.sql
```

Or run a specific query:
```bash
sqlite3 conversations.db "SELECT * FROM conversation_summary LIMIT 10;"
```

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- PyYAML >= 5.4.0
- pytest >= 7.0.0 (for testing)

## Documentation

- [Main README](README.md) - This file
- [Pipeline Documentation](PIPELINE_README.md) - Detailed pipeline guide
- [Tests](tests/) - Usage examples in test files
- [Configuration](config/pipeline_config.yaml) - Commented config file

## Issues Tracking

- **RUB-49**: Content Analysis Pipeline Implementation ✅ Complete

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License

## Author

Stephen Thompson

## Changelog

### v1.0.0 (2024)
- ✨ Added comprehensive content analysis pipeline (RUB-49)
- ✅ Relevance scoring with TF-IDF
- ✅ TextRank extractive summarization
- ✅ Duplicate detection with cosine similarity
- ✅ Priority classification system
- ✅ Main pipeline integration
- ✅ Configuration system
- ✅ Comprehensive test suite (90%+ coverage)
- 📚 Complete documentation
- 🚀 Setup and deployment scripts

### Earlier
- Basic conversation analysis scripts
- Database visualization tools
- SQL query templates
