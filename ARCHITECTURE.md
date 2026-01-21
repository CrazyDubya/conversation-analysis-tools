# Architecture Documentation

Detailed architecture and design patterns for conversation-analysis-tools.

## Table of Contents
1. [System Overview](#system-overview)
2. [Module Architecture](#module-architecture)
3. [Data Flow](#data-flow)
4. [Design Patterns](#design-patterns)
5. [Database Schema](#database-schema)
6. [Extension Points](#extension-points)

## System Overview

The conversation-analysis-tools system is designed with a modular architecture that separates concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CLI Tools  │  │  GUI Search  │  │   Pipeline   │     │
│  │ (Scripts)    │  │  (Tkinter)   │  │  (Batch)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────┬───────────┬───────────────┬───────────────┘
                 │           │               │
                 ▼           ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                      CORE MODULES                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Config    │  │   Database   │  │  SearchEngine│     │
│  │ (Settings)   │  │  (SQLite)    │  │  (Queries)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────┬──────────────────────────────-─┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANALYSIS PIPELINE                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Relevance   │  │  Summarizer  │  │  Duplicate   │     │
│  │   Scorer     │  │  (TextRank)  │  │  Detector    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐                                          │
│  │   Priority   │                                          │
│  │  Classifier  │                                          │
│  └──────────────┘                                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         SQLite Database (conversations.db)           │  │
│  │  ┌────────────────┐    ┌────────────────┐           │  │
│  │  │ conversations  │◄───┤   messages     │           │  │
│  │  └────────────────┘    └────────────────┘           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module Architecture

### Core Module Structure

The `core/` package provides foundational functionality:

```
core/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── database.py          # Database access layer
└── search_engine.py     # Search functionality

Responsibilities:
┌────────────────────────────────────────────────────────┐
│ core.config                                            │
│  • Load configuration from environment                 │
│  • Provide default values                              │
│  • Manage paths and directories                        │
│  • Platform color schemes                              │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ core.database                                          │
│  • Connection management (context manager)             │
│  • Common query patterns                               │
│  • Row factory configuration                           │
│  • Caching (platforms, models, date ranges)            │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ core.search_engine                                     │
│  • Keyword search with filters                         │
│  • Date range queries                                  │
│  • Platform/model filtering                            │
│  • Result export (CSV, JSON)                           │
└────────────────────────────────────────────────────────┘
```

### Pipeline Module Structure

The `pipeline/` package implements content analysis:

```
pipeline/
├── __init__.py
├── pipeline.py              # Main orchestrator
├── relevance_scorer.py      # TF-IDF relevance
├── summarizer.py            # Extractive summarization
├── duplicate_detector.py    # Similarity detection
└── priority_classifier.py   # Priority assignment

Flow:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Messages   │────▶│  Relevance   │────▶│  Filter by   │
│ from Database│     │    Scoring   │     │   Threshold  │
└──────────────┘     └──────────────┘     └──────────────┘
                             │
                             ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Summary    │◀────│ Summarizer   │◀────│  Relevant    │
│  Generation  │     │  (TextRank)  │     │   Messages   │
└──────────────┘     └──────────────┘     └──────────────┘
        │
        ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Priority   │────▶│  Duplicate   │────▶│   Final      │
│ Classification│     │  Detection   │     │  Results     │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Data Flow

### Search Request Flow

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  SearchEngine.keyword_search()  │
└─────────────────────────────────┘
    │
    ├─── Build SQL query with filters
    │
    ▼
┌─────────────────────────────────┐
│  DatabaseConnection.execute()   │
└─────────────────────────────────┘
    │
    ├─── Execute query
    │
    ▼
┌─────────────────────────────────┐
│  Convert to DataFrame           │
└─────────────────────────────────┘
    │
    ▼
Results returned to user
```

### Pipeline Processing Flow

```
Configuration
    │
    ▼
┌─────────────────────────────────┐
│  ContentAnalysisPipeline()      │
│  • Initialize modules           │
│  • Setup logging                │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  fetch_messages()               │
│  • Query database               │
│  • Apply platform filter        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  analyze_relevance()            │
│  • TF-IDF scoring               │
│  • Keyword matching             │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  generate_summaries()           │
│  • TextRank algorithm           │
│  • Extract key sentences        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  detect_duplicates()            │
│  • Cosine similarity            │
│  • Group similar messages       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  classify_priority()            │
│  • Multi-factor scoring         │
│  • Assign priority level        │
└─────────────────────────────────┘
    │
    ▼
Results
```

## Design Patterns

### 1. Context Manager Pattern

Used for automatic resource cleanup:

```python
# DatabaseConnection
with DatabaseConnection() as db:
    results = db.execute_query("SELECT ...")
# Connection automatically closed

# SearchEngine
with SearchEngine() as search:
    results = search.keyword_search("python")
# Connection automatically closed
```

**Benefits**:
- Automatic resource cleanup
- Exception-safe
- Prevents connection leaks

### 2. Configuration Pattern

Centralized configuration with environment override:

```
Environment Variables (.env)
           │
           ▼
     Config.DB_PATH ◄── Default: 'conversations.db'
           │
           ▼
    All modules use Config
```

**Benefits**:
- Single source of truth
- Easy environment-specific configuration
- No hardcoded values

### 3. Factory Pattern

Database row factory for consistent data access:

```python
conn.row_factory = sqlite3.Row
# Results are dict-like objects
result = cursor.fetchone()
value = result['column_name']  # Access by name
```

### 4. Strategy Pattern

Different analysis strategies in pipeline:

```
ContentAnalysisPipeline
    │
    ├─── RelevanceScorer (TF-IDF strategy)
    ├─── ExtractiveSummarizer (TextRank strategy)
    ├─── DuplicateDetector (Cosine similarity strategy)
    └─── PriorityClassifier (Multi-factor strategy)
```

**Benefits**:
- Swappable algorithms
- Easy to test
- Extensible

### 5. Caching Pattern

Performance optimization in database layer:

```python
class DatabaseConnection:
    def __init__(self):
        self._platforms_cache = None  # Lazy cache
    
    def get_platforms(self):
        if self._platforms_cache is None:
            # Query database once
            self._platforms_cache = self._query_platforms()
        return self._platforms_cache  # Return cached
```

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────────────────────┐
│        conversations            │
│─────────────────────────────────│
│ id (PK)          TEXT           │
│ title            TEXT           │
│ platform         TEXT           │
│ created_at       TIMESTAMP      │
│ updated_at       TIMESTAMP      │
│ account_id       TEXT           │
│ original_id      TEXT           │
│ metadata         TEXT (JSON)    │
└─────────────────────────────────┘
         │ 1
         │
         │ has many
         │
         ▼ *
┌─────────────────────────────────┐
│          messages               │
│─────────────────────────────────│
│ id (PK)          TEXT           │
│ conversation_id (FK) TEXT       │
│ parent_id        TEXT           │
│ sender           TEXT           │
│ role             TEXT           │
│ content          TEXT           │
│ created_at       TIMESTAMP      │
│ model            TEXT           │
│ order_index      INTEGER        │
│ metadata         TEXT (JSON)    │
└─────────────────────────────────┘
```

### Common Views

```sql
-- message_pairs: Join conversations with messages
CREATE VIEW message_pairs AS
SELECT 
    c.id as conversation_id,
    c.platform,
    m.id as message_id,
    m.sender,
    m.content
FROM conversations c
JOIN messages m ON c.id = m.conversation_id;

-- conversation_summary: Aggregate statistics
CREATE VIEW conversation_summary AS
SELECT 
    c.id,
    c.platform,
    COUNT(m.id) as message_count,
    MIN(m.created_at) as first_message,
    MAX(m.created_at) as last_message
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
GROUP BY c.id;
```

## Extension Points

### Adding a New Search Method

1. Add method to `SearchEngine` class:
```python
def custom_search(self, criteria: dict) -> pd.DataFrame:
    """Your custom search logic."""
    sql_query = self._build_custom_query(criteria)
    results = self.db.execute_query(sql_query)
    return pd.DataFrame([dict(row) for row in results])
```

2. Use it:
```python
with SearchEngine() as search:
    results = search.custom_search({'min_length': 100})
```

### Adding a New Pipeline Module

1. Create new module in `pipeline/`:
```python
class CustomAnalyzer:
    def analyze(self, documents: List[str]) -> List[dict]:
        """Your analysis logic."""
        return results
```

2. Integrate in pipeline:
```python
class ContentAnalysisPipeline:
    def __init__(self):
        self.custom_analyzer = CustomAnalyzer()
    
    def process(self):
        # ... existing code ...
        custom_results = self.custom_analyzer.analyze(messages)
```

### Adding Configuration Options

1. Add to `Config` class:
```python
class Config:
    NEW_OPTION: str = os.getenv('NEW_OPTION', 'default_value')
```

2. Add to `.env.example`:
```bash
# New feature configuration
NEW_OPTION=your_value
```

3. Use in code:
```python
from core.config import Config
value = Config.NEW_OPTION
```

### Adding a New Data Source

1. Create parser in `uni_parse.py`:
```python
def _process_newsource_file(self, file_path: str):
    """Parse new source format."""
    # Your parsing logic
    pass
```

2. Add platform detection:
```python
def _detect_platform(self, file_path: str) -> str:
    # ... existing code ...
    elif "newsource_marker" in first_chars:
        return "newsource"
```

3. Update schema if needed and create migration.

## Performance Considerations

### Database Indexes

Recommended indexes for optimal performance:

```sql
-- Message content search
CREATE INDEX idx_messages_content ON messages(content);

-- Time-based queries
CREATE INDEX idx_messages_created ON messages(created_at);
CREATE INDEX idx_conversations_created ON conversations(created_at);

-- Platform filtering
CREATE INDEX idx_conversations_platform ON conversations(platform);

-- Foreign key optimization
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
```

### Caching Strategy

```
┌─────────────────────────────────┐
│  Application Layer              │
│  • Python object caching        │
│  • pandas DataFrame caching     │
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  Database Layer                 │
│  • Query result caching         │
│  • Connection pooling           │
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  SQLite Layer                   │
│  • Page cache                   │
│  • Query planner cache          │
└─────────────────────────────────┘
```

### Memory Management

For large datasets:

```
Small Dataset (<10K messages)
    └─→ Load all into memory

Medium Dataset (10K-100K messages)
    └─→ Process in batches

Large Dataset (>100K messages)
    └─→ Use generators/iterators
```

## Security Considerations

1. **SQL Injection Prevention**: Always use parameterized queries
   ```python
   # Good
   db.execute_query("SELECT * FROM messages WHERE id = ?", (message_id,))
   
   # Bad - vulnerable to SQL injection
   db.execute_query(f"SELECT * FROM messages WHERE id = '{message_id}'")
   ```

2. **Path Traversal Prevention**: Validate file paths
   ```python
   from pathlib import Path
   safe_path = Path(db_path).resolve()
   ```

3. **Data Validation**: Validate inputs before processing
   ```python
   if not isinstance(limit, int) or limit < 1:
       raise ValueError("Limit must be positive integer")
   ```

## Testing Architecture

```
tests/
├── test_config.py           # Unit tests for Config
├── test_database.py         # Unit tests for DatabaseConnection
├── test_search_engine.py    # Unit tests for SearchEngine
├── test_pipeline.py         # Integration tests
├── test_uni_parse.py        # Parser tests
└── conftest.py              # Shared fixtures

Test Strategy:
├── Unit Tests (Fast, Isolated)
│   └─→ Test individual functions
├── Integration Tests (Medium)
│   └─→ Test module interactions
└── End-to-End Tests (Slow)
    └─→ Test complete workflows
```

## Deployment Architecture

```
Development
    │
    ├─→ Local SQLite database
    ├─→ Virtual environment
    └─→ Development dependencies
    
Production
    │
    ├─→ Centralized database
    ├─→ Environment variables for config
    └─→ Production dependencies only

CI/CD
    │
    ├─→ GitHub Actions
    ├─→ Automated testing
    ├─→ Code quality checks
    └─→ Coverage reporting
```

## Future Enhancements

Planned architectural improvements:

1. **Connection Pooling**: For concurrent access
2. **Async Support**: For I/O-bound operations
3. **Plugin System**: For third-party extensions
4. **API Layer**: RESTful API using FastAPI
5. **Distributed Processing**: For large-scale analysis

---

For more details, see:
- [CODE_REVIEW.md](CODE_REVIEW.md) - Code quality analysis
- [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - Technical improvements
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
