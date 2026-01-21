# Deprecated Files

This directory contains legacy files that have been superseded by the new core modules. These files are kept for reference during the migration period.

## Deprecated Files

| File | Lines | Reason | Replacement |
|------|-------|--------|-------------|
| `exper_sql.py` | 2,724 | Monolithic, mixed concerns | `core/database.py`, `core/search_engine.py` |
| `sql_search.py` | 2,246 | Duplicate search logic | `core/search_engine.py` |
| `gui_sql.py` | 1,139 | Duplicate search/DB logic | `core/database.py`, `core/search_engine.py` |

## Migration Guide

### Old Pattern (Deprecated)
```python
# OLD: Using exper_sql.py
from exper_sql import ConversationAnalyzer

DB_PATH = "/Users/pup/Desktop/Arch/conversations.db"
analyzer = ConversationAnalyzer(DB_PATH)
# ... use analyzer methods
```

### New Pattern (Recommended)
```python
# NEW: Using core modules
from core.config import Config
from core.database import DatabaseConnection
from core.search_engine import SearchEngine

# Configuration is centralized and environment-aware
config = Config()

# Use with context managers for automatic cleanup
with DatabaseConnection() as db:
    platforms = db.get_platforms()
    results = db.execute_query("SELECT * FROM conversations LIMIT 10")

with SearchEngine() as search:
    results = search.keyword_search("machine learning", platform="claude")
    search.export_results(results, "search_results", format='csv')
```

## Benefits of Core Modules

1. **Centralized Configuration**: No more hardcoded paths
2. **Reduced Duplication**: Single source of truth for DB and search logic
3. **Better Testing**: Modular design enables unit testing
4. **Environment Support**: Use `.env` files for different environments
5. **Consistent Interface**: Same API across all tools
6. **Improved Logging**: Built-in logging for debugging

## Timeline

- **Phase 1** (Week 1-2): Core modules created, main files migrated
- **Phase 2** (Week 3): Testing and validation
- **Phase 3** (Week 4): Documentation updates
- **Removal**: Deprecated files will be removed after 2 releases

## Need Help?

See [REFACTORING_PLAN.md](../REFACTORING_PLAN.md) for details or [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute to the migration.

---

**Last Updated**: 2026-01-21
