# Coding Standards

Code style and quality standards for conversation-analysis-tools.

## Table of Contents
1. [General Principles](#general-principles)
2. [Python Style Guide](#python-style-guide)
3. [Documentation Standards](#documentation-standards)
4. [Testing Standards](#testing-standards)
5. [Error Handling](#error-handling)
6. [Performance Guidelines](#performance-guidelines)
7. [Security Guidelines](#security-guidelines)

## General Principles

### Code Quality Pillars

1. **Readability** - Code is read more than written
2. **Maintainability** - Easy to modify and extend
3. **Testability** - Easy to test in isolation
4. **Performance** - Efficient but not premature optimization
5. **Security** - Safe from common vulnerabilities

### Design Principles

- **DRY (Don't Repeat Yourself)**: Eliminate duplication
- **KISS (Keep It Simple, Stupid)**: Prefer simple solutions
- **YAGNI (You Aren't Gonna Need It)**: Don't add unnecessary features
- **Separation of Concerns**: Each module has a single responsibility
- **Fail Fast**: Validate inputs early and fail with clear errors

## Python Style Guide

### Base Standard

Follow **PEP 8** with these modifications:
- Line length: Maximum **100 characters**
- String quotes: Prefer **double quotes** for user-facing strings
- Imports: Group and sort with **isort**

### Code Formatting

Use **Black** for automatic formatting:

```bash
black . --line-length=100
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
```

### Import Organization

Use **isort** with Black-compatible settings:

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Local imports
from core.config import Config
from core.database import DatabaseConnection
```

Configuration:
```bash
isort . --profile black --line-length=100
```

### Type Hints

**Required** for all public functions and methods:

```python
def search_messages(
    self,
    query: str,
    platform: Optional[str] = None,
    limit: int = 100
) -> pd.DataFrame:
    """Search messages with optional filters."""
    pass
```

**Use modern type hint syntax**:
```python
# Good - Python 3.9+
from typing import Optional, List, Dict

def process(data: List[str]) -> Dict[str, int]:
    pass

# Better - Python 3.10+
def process(data: list[str]) -> dict[str, int]:
    pass
```

### Naming Conventions

Follow PEP 8 naming:

| Type | Convention | Example |
|------|-----------|---------|
| Module | `lowercase_with_underscores` | `search_engine.py` |
| Class | `PascalCase` | `DatabaseConnection` |
| Function | `lowercase_with_underscores` | `get_platforms()` |
| Constant | `UPPERCASE_WITH_UNDERSCORES` | `DEFAULT_PLATFORM` |
| Private | `_leading_underscore` | `_build_query()` |

### Function/Method Length

- **Ideal**: < 20 lines
- **Maximum**: < 50 lines
- **Over 50 lines**: Consider refactoring into smaller functions

### Class Design

- **Single Responsibility**: Each class should have one clear purpose
- **Small Classes**: Prefer many small classes over few large ones
- **Composition over Inheritance**: Use composition when possible

```python
# Good - composition
class SearchEngine:
    def __init__(self):
        self.db = DatabaseConnection()  # Composition
        self.config = Config()

# Avoid deep inheritance hierarchies
```

## Documentation Standards

### Docstring Format

Use **Google-style** or **NumPy-style** docstrings:

```python
def keyword_search(
    self,
    query: str,
    platform: Optional[str] = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    Perform keyword search across messages.
    
    Args:
        query: Search keywords
        platform: Filter by platform ('claude', 'chatgpt', or None for all)
        limit: Maximum results to return (default: 100)
        
    Returns:
        DataFrame with search results containing columns:
        - message_id: Unique message identifier
        - conversation_id: Parent conversation ID
        - content: Message content
        - platform: Platform name
        - created_at: Timestamp
        
    Raises:
        ValueError: If limit is not positive
        RuntimeError: If database is not connected
        
    Example:
        >>> with SearchEngine() as search:
        ...     results = search.keyword_search("python", platform="claude")
        ...     print(f"Found {len(results)} results")
    """
    pass
```

### Module Docstrings

Every module should have a docstring:

```python
"""
Core configuration management module.

This module provides centralized configuration with environment variable
support, replacing hardcoded paths throughout the codebase.

Usage:
    from core.config import Config
    
    config = Config()
    db_path = config.DB_PATH
"""
```

### Comments

- **When to comment**: Explain *why*, not *what*
- **Avoid obvious comments**: Code should be self-documenting
- **Keep comments updated**: Outdated comments are worse than no comments

```python
# Bad - obvious comment
# Increment counter by 1
counter += 1

# Good - explains why
# Add 1 to account for zero-indexing
counter += 1

# Better - self-documenting code
counter = zero_indexed_count + 1
```

## Testing Standards

### Test Structure

Follow **Arrange-Act-Assert** (AAA) pattern:

```python
def test_keyword_search_with_platform_filter():
    """Test keyword search filters by platform correctly."""
    # Arrange - set up test data
    with SearchEngine(test_db_path) as search:
        
        # Act - perform the operation
        results = search.keyword_search("test", platform="claude")
        
        # Assert - verify results
        assert len(results) > 0
        assert all(results['platform'] == 'claude')
```

### Test Naming

Use descriptive names that explain what is being tested:

```python
# Good
def test_search_returns_empty_dataframe_when_no_matches():
    pass

def test_duplicate_detector_groups_similar_messages():
    pass

# Bad
def test_search():
    pass

def test1():
    pass
```

### Test Coverage

- **Minimum**: 70% coverage for new code
- **Target**: 80%+ coverage for core modules
- **Focus**: Test edge cases and error conditions

### Test Fixtures

Use fixtures for reusable test data:

```python
@pytest.fixture
def sample_messages():
    """Provide sample message data for tests."""
    return [
        {'id': 'msg1', 'content': 'Test message 1'},
        {'id': 'msg2', 'content': 'Test message 2'},
    ]

def test_relevance_scoring(sample_messages):
    """Test scoring uses fixture data."""
    scorer = RelevanceScorer()
    scores = scorer.score_documents(sample_messages)
    assert len(scores) == len(sample_messages)
```

## Error Handling

### Exception Handling

- **Be specific**: Catch specific exceptions, not `Exception`
- **Provide context**: Include helpful error messages
- **Fail fast**: Validate inputs early

```python
# Good
def get_conversation(self, conversation_id: str) -> dict:
    """Get conversation by ID."""
    if not conversation_id:
        raise ValueError("conversation_id cannot be empty")
    
    try:
        result = self.db.execute_query(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,)
        )
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to fetch conversation {conversation_id}: {e}")
    
    if not result:
        raise NotFoundError(f"Conversation not found: {conversation_id}")
    
    return dict(result[0])

# Bad
def get_conversation(self, conversation_id):
    try:
        result = self.db.execute_query(
            f"SELECT * FROM conversations WHERE id = '{conversation_id}'"
        )
        return result[0]
    except:
        return None
```

### Custom Exceptions

Define custom exceptions for domain-specific errors:

```python
class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass

class SearchError(Exception):
    """Raised when search operations fail."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass
```

### Error Messages

Write clear, actionable error messages:

```python
# Good
raise ValueError(
    f"Invalid platform: '{platform}'. "
    f"Must be one of: {', '.join(valid_platforms)}"
)

# Bad
raise ValueError("Invalid platform")
```

## Performance Guidelines

### Database Queries

- **Use parameterized queries**: Prevent SQL injection, enable query caching
- **Add appropriate indexes**: For frequently queried columns
- **Limit results**: Use LIMIT clause to avoid loading unnecessary data
- **Use transactions**: For multiple related operations

```python
# Good - parameterized, limited
results = db.execute_query(
    "SELECT * FROM messages WHERE platform = ? LIMIT ?",
    (platform, limit)
)

# Bad - vulnerable, unlimited
results = db.execute_query(
    f"SELECT * FROM messages WHERE platform = '{platform}'"
)
```

### Memory Management

- **Use generators**: For processing large datasets
- **Process in batches**: Don't load everything into memory
- **Clean up resources**: Use context managers

```python
# Good - generator
def process_messages():
    with DatabaseConnection() as db:
        for row in db.execute_query("SELECT * FROM messages"):
            yield process_message(row)

# Bad - loads all into memory
def process_messages():
    db = DatabaseConnection()
    all_messages = list(db.execute_query("SELECT * FROM messages"))
    return [process_message(m) for m in all_messages]
```

### Caching

- **Cache expensive computations**: Store results of slow operations
- **Invalidate appropriately**: Clear cache when data changes
- **Consider memory**: Don't cache everything

```python
class DataService:
    def __init__(self):
        self._cache = {}
    
    def get_expensive_data(self, key: str):
        if key not in self._cache:
            self._cache[key] = self._compute_expensive(key)
        return self._cache[key]
```

## Security Guidelines

### SQL Injection Prevention

**Always** use parameterized queries:

```python
# Good - parameterized
db.execute_query(
    "SELECT * FROM messages WHERE id = ?",
    (message_id,)
)

# Bad - vulnerable to SQL injection
db.execute_query(
    f"SELECT * FROM messages WHERE id = '{message_id}'"
)
```

### Path Traversal Prevention

Validate and sanitize file paths:

```python
from pathlib import Path

def safe_read_file(file_path: str) -> str:
    """Safely read file, preventing path traversal."""
    # Resolve to absolute path
    safe_path = Path(file_path).resolve()
    
    # Ensure it's within allowed directory
    allowed_dir = Path('/allowed/directory').resolve()
    if not str(safe_path).startswith(str(allowed_dir)):
        raise ValueError(f"Path outside allowed directory: {file_path}")
    
    return safe_path.read_text()
```

### Input Validation

Validate all external inputs:

```python
def process_limit(limit: int) -> int:
    """Validate and sanitize limit parameter."""
    if not isinstance(limit, int):
        raise TypeError(f"Limit must be integer, got {type(limit)}")
    
    if limit < 1:
        raise ValueError(f"Limit must be positive, got {limit}")
    
    if limit > 10000:
        raise ValueError(f"Limit too large (max 10000), got {limit}")
    
    return limit
```

### Secrets Management

Never commit secrets:

```python
# Good - use environment variables
import os
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY environment variable not set")

# Bad - hardcoded secret
api_key = "sk-1234567890abcdef"  # Never do this!
```

## Code Review Checklist

Before submitting code for review, ensure:

- [ ] Code follows PEP 8 and Black formatting
- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] Tests added for new functionality
- [ ] No hardcoded values (use Config)
- [ ] Error handling is appropriate
- [ ] No SQL injection vulnerabilities
- [ ] Performance is acceptable
- [ ] Documentation is updated
- [ ] Commit messages are clear

## Tools and Automation

### Pre-commit Hooks

Install and use pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This automatically runs:
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Continuous Integration

All code must pass CI checks:
- Unit tests (pytest)
- Code quality (flake8, black, isort)
- Type checking (mypy)
- Coverage reporting

### IDE Configuration

Recommended VSCode settings:

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=100"],
    "editor.formatOnSave": true,
    "python.linting.mypyEnabled": true
}
```

## References

- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [PEP 257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
- [Black Code Style](https://black.readthedocs.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [CONTRIBUTING.md](CONTRIBUTING.md) - Project contribution guide

---

**Last Updated**: 2026-01-21
