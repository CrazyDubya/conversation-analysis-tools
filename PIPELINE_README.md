# Content Analysis Pipeline

A comprehensive NLP pipeline for analyzing scraped research content from Claude and ChatGPT conversations.

## Features

### 1. Relevance Scoring System
- **TF-IDF based relevance** calculation
- **Keyword density** analysis
- **Keyword coverage** metrics
- **Multi-criteria scoring** with configurable weights
- Corpus-wide IDF caching for efficiency

### 2. Extractive Summarization
- **TextRank algorithm** for sentence importance
- **Configurable summary length**
- Support for **multi-document summarization**
- Sentence similarity using word overlap
- PageRank-style iterative scoring

### 3. Duplicate Detection
- **Cosine similarity** for content comparison
- **Configurable similarity threshold**
- Efficient pairwise comparison
- **Document clustering** for duplicate groups
- Unique document extraction

### 4. Priority Classification
- **Multi-factor priority scoring**:
  - Relevance score
  - Document length
  - Keyword matching
  - Recency (optional)
- **Five priority levels**: CRITICAL, HIGH, MEDIUM, LOW, NONE
- Configurable thresholds and weights
- Batch processing support

### 5. Main Pipeline Integration
- **End-to-end processing** workflow
- Database integration (SQLite)
- Batch processing
- Results aggregation
- JSON output export
- Database storage of results
- Comprehensive logging

## Installation

### Quick Setup

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p output logs
```

### Install as Package

```bash
pip install -e .
```

## Configuration

Edit `config/pipeline_config.yaml` to customize:

```yaml
# Keywords for relevance scoring
keywords:
  - machine learning
  - deep learning
  - artificial intelligence
  # ... more keywords

# Relevance scoring weights
relevance:
  weights:
    density: 0.3      # Keyword density weight
    coverage: 0.4     # Keyword coverage weight
    tfidf: 0.3        # TF-IDF weight

# Summarization settings
summarizer:
  damping: 0.85               # PageRank damping factor
  summary_sentences: 3         # Default sentences to extract

# Duplicate detection threshold
duplicate_threshold: 0.8      # Cosine similarity (0-1)

# Priority classification
priority:
  relevance_thresholds:
    critical: 0.9
    high: 0.7
    medium: 0.5
    low: 0.3
  
  length_thresholds:
    min_words: 50
    substantial: 200
    comprehensive: 500
  
  keyword_thresholds:
    critical_keywords:
      - breakthrough
      - novel
      - significant
    high_keywords:
      - important
      - effective
      - advanced
```

## Usage

### Command Line

```bash
# Run with default settings
python run_pipeline.py

# Specify custom config
python run_pipeline.py --config my_config.yaml

# Analyze specific platform
python run_pipeline.py --platform claude --limit 100

# Save to custom location
python run_pipeline.py --output results/my_analysis.json

# Skip database storage
python run_pipeline.py --no-save-db
```

### Python API

```python
from pipeline import ContentAnalysisPipeline
import yaml

# Load configuration
with open('config/pipeline_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = ContentAnalysisPipeline(
    config=config,
    db_path='conversations.db'
)

# Run complete analysis
results = pipeline.process(
    platform='claude',     # or 'chatgpt' or None for all
    limit=100,             # max messages to process
    skip_duplicates=True   # filter duplicates
)

# Save results
pipeline.save_results(results, 'output/results.json')
pipeline.store_results_db(results)

# Access specific results
for doc_id, doc_data in results['documents'].items():
    print(f"Document {doc_id}:")
    print(f"  Relevance: {doc_data['relevance']['combined']:.3f}")
    print(f"  Priority: {doc_data['priority']['level']}")
    print(f"  Summary: {doc_data['summary']}")
    print()
```

### Individual Modules

```python
from pipeline import (
    RelevanceScorer,
    ExtractiveSummarizer,
    DuplicateDetector,
    PriorityClassifier,
    PriorityLevel
)

# Relevance Scoring
scorer = RelevanceScorer(keywords=['AI', 'machine learning'])
documents = [(1, "Text about AI..."), (2, "More text...")]
ranked = scorer.rank_documents(documents, top_k=10)

for doc_id, score, details in ranked:
    print(f"Doc {doc_id}: {score:.3f}")

# Summarization
summarizer = ExtractiveSummarizer()
text = "Long document text here..."
summary = summarizer.summarize_to_text(text, num_sentences=3)
print(summary)

# Duplicate Detection
detector = DuplicateDetector(similarity_threshold=0.8)
duplicates = detector.find_duplicates(documents)
print(f"Found {len(duplicates)} duplicate pairs")

# Get unique documents
unique_ids = detector.get_unique_documents(documents)
print(f"Unique documents: {unique_ids}")

# Priority Classification
classifier = PriorityClassifier()
docs_with_scores = [(1, "Text...", 0.8), (2, "More text...", 0.6)]
ranked = classifier.rank_by_priority(docs_with_scores)

for doc_id, priority in ranked:
    print(f"Doc {doc_id}: {priority.level.name} ({priority.score:.3f})")
    print(f"  Reasons: {', '.join(priority.reasons[:2])}")
```

## Results Format

The pipeline generates a JSON file with the following structure:

```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:00",
    "platform": "claude",
    "total_documents": 50,
    "duplicate_pairs": 5,
    "processing_time": 2.34
  },
  "documents": {
    "1": {
      "text": "Document text (truncated)...",
      "relevance": {
        "combined": 0.87,
        "density": 0.12,
        "coverage": 0.85,
        "tfidf": 0.91
      },
      "summary": "Key sentence 1. Key sentence 2. Key sentence 3.",
      "priority": {
        "level": "HIGH",
        "score": 0.73,
        "reasons": [
          "Overall priority score: 0.73",
          "Substantial content (200+ words)",
          "High relevance score (0.87)"
        ],
        "metadata": {
          "relevance_score": 0.87,
          "length_score": 0.8,
          "keyword_score": 0.7,
          "recency_score": 0.5
        }
      }
    }
  },
  "duplicates": [
    [1, 5, 0.95],  // [id1, id2, similarity]
    [3, 8, 0.89]
  ],
  "statistics": {
    "relevance": {
      "average": 0.65,
      "min": 0.23,
      "max": 0.94
    },
    "priority_distribution": {
      "CRITICAL": 2,
      "HIGH": 15,
      "MEDIUM": 20,
      "LOW": 10,
      "NONE": 3
    }
  }
}
```

## Database Schema

Results are stored in the `analysis_results` table:

```sql
CREATE TABLE analysis_results (
    message_id INTEGER PRIMARY KEY,
    relevance_score REAL,
    summary TEXT,
    priority_level TEXT,
    priority_score REAL,
    analyzed_at TEXT,
    metadata TEXT  -- JSON
);
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pipeline --cov-report=html

# Run specific test file
pytest tests/test_relevance_scorer.py -v

# Run specific test
pytest tests/test_pipeline.py::TestContentAnalysisPipeline::test_process_full_pipeline -v
```

## Performance

### Benchmarks (on 1000 documents)

- **Relevance Scoring**: ~2-3 seconds
- **Summarization**: ~5-8 seconds
- **Duplicate Detection**: ~3-5 seconds
- **Priority Classification**: ~1-2 seconds
- **Full Pipeline**: ~12-18 seconds

### Optimization Tips

1. **Use batch processing** for large datasets
2. **Adjust duplicate threshold** (higher = faster)
3. **Limit summary sentences** (fewer = faster)
4. **Process by platform** separately
5. **Cache IDF values** for repeated analyses

## Architecture

```
pipeline/
├── __init__.py              # Package exports
├── relevance_scorer.py      # TF-IDF relevance scoring
├── summarizer.py            # TextRank summarization
├── duplicate_detector.py    # Cosine similarity detection
├── priority_classifier.py   # Multi-factor classification
└── pipeline.py              # Main integration class

config/
└── pipeline_config.yaml     # Configuration file

tests/
├── conftest.py             # Fixtures
├── test_relevance_scorer.py
├── test_summarizer.py
├── test_duplicate_detector.py
├── test_priority_classifier.py
└── test_pipeline.py        # Integration tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - See LICENSE file

## References

- **TextRank**: Mihalcea & Tarau (2004)
- **TF-IDF**: Salton & McGill (1986)
- **PageRank**: Page et al. (1999)
- **Cosine Similarity**: Salton & McGill (1986)

## Troubleshooting

### Issue: "No module named 'pipeline'"
**Solution**: Install the package with `pip install -e .` or ensure the directory is in PYTHONPATH.

### Issue: "Database file not found"
**Solution**: Specify correct path with `--db path/to/conversations.db`

### Issue: "Out of memory"
**Solution**: Process in smaller batches using `--limit` parameter

### Issue: "Slow performance"
**Solution**: 
- Reduce `duplicate_threshold` to skip fewer comparisons
- Decrease `summary_sentences`
- Process fewer documents with `--limit`

## Support

For issues and questions:
- Open an issue on GitHub
- Check the test files for usage examples
- Review the configuration comments
