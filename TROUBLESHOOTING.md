# Troubleshooting Guide

Common issues and solutions for conversation-analysis-tools.

## Installation Issues

### Issue: Module not found errors

**Problem**: `ModuleNotFoundError: No module named 'core'` or similar

**Solutions**:
1. Ensure you're in the correct directory:
   ```bash
   cd /path/to/conversation-analysis-tools
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. If using the core modules, ensure you're running from the repository root:
   ```bash
   python -m pytest tests/
   # NOT: cd tests && pytest
   ```

### Issue: Database file not found

**Problem**: `FileNotFoundError: Database file not found: conversations.db`

**Solutions**:
1. Set DB_PATH environment variable:
   ```bash
   export DB_PATH=/path/to/your/conversations.db
   ```

2. Or create a `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and set DB_PATH=your/path/to/conversations.db
   ```

3. Or use default location:
   ```bash
   # Place conversations.db in the repository root
   ls conversations.db  # Should exist
   ```

### Issue: Python version compatibility

**Problem**: `SyntaxError` or incompatible type hints

**Solutions**:
1. Check Python version (requires 3.8+):
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. Create virtual environment with correct version:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Database Issues

### Issue: Views not found

**Problem**: `WARNING: Missing database views: message_pairs, conversation_summary`

**Solution**: Create the views using the SQL script:
```bash
sqlite3 conversations.db < create_views.sql
```

### Issue: Database locked

**Problem**: `sqlite3.OperationalError: database is locked`

**Solutions**:
1. Ensure no other processes are using the database:
   ```bash
   lsof conversations.db  # Check open file handles
   ```

2. Use context managers for proper connection cleanup:
   ```python
   # Good - automatically closes
   with DatabaseConnection() as db:
       results = db.execute_query("SELECT * FROM conversations")
   
   # Bad - may leave connection open
   db = DatabaseConnection()
   db.connect()
   # ... forgot to call db.close()
   ```

3. Set a timeout:
   ```python
   conn = sqlite3.connect(db_path, timeout=10.0)
   ```

### Issue: Foreign key constraints

**Problem**: `IntegrityError: FOREIGN KEY constraint failed`

**Solution**: Ensure conversation exists before adding messages:
```python
# Create conversation first
db.execute_update(
    "INSERT INTO conversations (id, title, platform) VALUES (?, ?, ?)",
    ('conv-123', 'My Conversation', 'claude')
)

# Then add messages
db.execute_update(
    "INSERT INTO messages (id, conversation_id, content) VALUES (?, ?, ?)",
    ('msg-1', 'conv-123', 'Hello')
)
```

## Search Issues

### Issue: No search results

**Problem**: Search returns empty DataFrame

**Solutions**:
1. Verify data exists in database:
   ```python
   with DatabaseConnection() as db:
       stats = db.execute_query("SELECT COUNT(*) as count FROM messages")
       print(f"Total messages: {stats[0]['count']}")
   ```

2. Check search parameters:
   ```python
   # Case-sensitive database - search is case-insensitive via LIKE
   results = search.keyword_search("python")  # Finds "Python", "python", "PYTHON"
   ```

3. Try broader search:
   ```python
   # Remove filters
   results = search.keyword_search("machine learning", platform=None, sender=None)
   ```

### Issue: Search too slow

**Problem**: Queries taking too long

**Solutions**:
1. Add indexes to database:
   ```sql
   CREATE INDEX IF NOT EXISTS idx_messages_content ON messages(content);
   CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
   CREATE INDEX IF NOT EXISTS idx_conversations_platform ON conversations(platform);
   ```

2. Limit results:
   ```python
   results = search.keyword_search("python", limit=50)  # Instead of 1000
   ```

3. Use more specific queries:
   ```python
   # Good - specific platform
   results = search.keyword_search("python", platform="claude")
   
   # Less efficient - searches all platforms
   results = search.keyword_search("python")
   ```

## Pipeline Issues

### Issue: Pipeline configuration errors

**Problem**: `KeyError` or `AttributeError` in pipeline config

**Solutions**:
1. Check config file exists and is valid YAML:
   ```bash
   python -c "import yaml; yaml.safe_load(open('config/pipeline_config.yaml'))"
   ```

2. Use default configuration:
   ```python
   from pipeline import ContentAnalysisPipeline
   
   # Uses defaults if config not provided
   pipeline = ContentAnalysisPipeline()
   ```

3. Validate required keys:
   ```python
   import yaml
   config = yaml.safe_load(open('config/pipeline_config.yaml'))
   required_keys = ['keywords', 'relevance', 'summarizer', 'priority']
   missing = [k for k in required_keys if k not in config]
   if missing:
       print(f"Missing config keys: {missing}")
   ```

### Issue: Out of memory errors

**Problem**: `MemoryError` when processing large datasets

**Solutions**:
1. Process in batches:
   ```python
   pipeline = ContentAnalysisPipeline(db_path='conversations.db')
   
   # Instead of processing all at once
   results = pipeline.process(limit=100)  # Process 100 at a time
   ```

2. Use generators for streaming:
   ```python
   def process_in_batches(batch_size=100):
       offset = 0
       while True:
           results = pipeline.process(limit=batch_size, offset=offset)
           if not results:
               break
           yield results
           offset += batch_size
   ```

## Test Issues

### Issue: Tests fail with "No module named 'pytest'"

**Problem**: Test dependencies not installed

**Solution**: Install dev dependencies:
```bash
pip install -r requirements-dev.txt
```

### Issue: Database fixture errors

**Problem**: `PermissionError` or fixture cleanup issues

**Solutions**:
1. Ensure temp directory is writable:
   ```bash
   ls -la /tmp  # Check permissions
   ```

2. Clean up manually:
   ```bash
   rm -f /tmp/test*.db
   ```

3. Use pytest with proper cleanup:
   ```bash
   pytest tests/ -v --tb=short
   ```

## Import Issues

### Issue: Circular import errors

**Problem**: `ImportError: cannot import name 'X' from partially initialized module`

**Solution**: The core modules are designed to avoid circular imports. Ensure you're importing correctly:
```python
# Good - import from core modules
from core.config import Config
from core.database import DatabaseConnection
from core.search_engine import SearchEngine

# Bad - don't import deprecated modules with core modules
from deprecated.exper_sql import ConversationAnalyzer  # Don't mix!
```

### Issue: Relative import errors

**Problem**: `ImportError: attempted relative import with no known parent package`

**Solutions**:
1. Run as module from project root:
   ```bash
   python -m pipeline.pipeline  # Good
   # NOT: cd pipeline && python pipeline.py  # Bad
   ```

2. Or add project to PYTHONPATH:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/conversation-analysis-tools"
   ```

## Configuration Issues

### Issue: Environment variables not loaded

**Problem**: Config uses default values instead of .env values

**Solutions**:
1. Install python-dotenv:
   ```bash
   pip install python-dotenv
   ```

2. Load environment file explicitly:
   ```python
   from core.config import Config
   config = Config.load_from_env('.env')  # Explicitly load
   ```

3. Or use shell environment:
   ```bash
   export DB_PATH=/my/path/conversations.db
   python my_script.py
   ```

### Issue: Paths not working on Windows

**Problem**: `FileNotFoundError` with backslash paths

**Solution**: Use pathlib for cross-platform compatibility:
```python
from pathlib import Path

# Good - works on all platforms
db_path = Path('conversations.db')
output_dir = Path('output')

# Config already uses pathlib internally
from core.config import Config
print(Config.OUTPUT_DIR)  # Already a Path object
```

## Performance Issues

### Issue: Slow startup time

**Problem**: Scripts take long to start

**Solutions**:
1. Lazy import heavy libraries:
   ```python
   # At top of file - fast
   import sqlite3
   
   # Only when needed - deferred
   def visualize():
       import matplotlib.pyplot as plt  # Import only when needed
       plt.plot(data)
   ```

2. Use connection pooling (for long-running apps):
   ```python
   # Keep connection alive for multiple queries
   with DatabaseConnection() as db:
       for query in queries:
           results = db.execute_query(query)
   ```

### Issue: High memory usage

**Problem**: Process uses too much memory

**Solutions**:
1. Use iterators instead of lists:
   ```python
   # Bad - loads everything into memory
   messages = list(db.execute_query("SELECT * FROM messages"))
   
   # Good - process one at a time
   for row in db.execute_query("SELECT * FROM messages"):
       process(row)
   ```

2. Clean up large objects:
   ```python
   import gc
   results = heavy_computation()
   process(results)
   del results  # Free memory
   gc.collect()
   ```

## Getting Help

If you're still experiencing issues:

1. **Check the logs**: Look for detailed error messages in `logs/` directory

2. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Search existing issues**: [GitHub Issues](https://github.com/CrazyDubya/conversation-analysis-tools/issues)

4. **Ask for help**: Open a new issue with:
   - Python version (`python --version`)
   - Error message and full traceback
   - Minimal code to reproduce
   - What you've already tried

5. **Check documentation**:
   - [README.md](README.md) - Getting started
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
   - [CODE_REVIEW.md](CODE_REVIEW.md) - Architecture details
   - [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - Technical details
