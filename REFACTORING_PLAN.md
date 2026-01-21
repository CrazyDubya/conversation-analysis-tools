# Refactoring Plan for Conversation Analysis Tools

**Created**: 2026-01-21  
**Status**: Planning Phase  
**Reference**: See [CODE_REVIEW.md](CODE_REVIEW.md) for detailed analysis

---

## 📋 Overview

This document outlines the refactoring strategy to address technical debt identified in the comprehensive code review. The plan prioritizes high-impact, manageable changes that will improve code maintainability, testability, and quality.

---

## 🎯 Goals

1. **Reduce code duplication** from ~30% to <5%
2. **Improve test coverage** from 65% to >80%
3. **Add type safety** from 10% to >80% type hint coverage
4. **Modularize monolithic files** (split 2,000+ line files)
5. **Centralize configuration** (eliminate hardcoded paths)

---

## 🔴 Priority 0: Critical Issues (Sprint 1)

### 1. Split Monolithic `exper_sql.py` (1,996 lines)

**Current State**:
- Single file with 84 functions
- Mixes database access, UI, and business logic
- Difficult to test and maintain

**Target Architecture**:
```
database/
  ├── __init__.py
  ├── db_access.py         # Database connection and queries
  ├── db_models.py         # Data models and schemas
  └── db_queries.py        # Reusable query functions

ui/
  ├── __init__.py
  └── search_interface.py  # UI components

analysis/
  ├── __init__.py
  └── sql_analyzer.py      # Analysis logic
```

**Implementation Steps**:
1. [ ] Create new directory structure
2. [ ] Extract database connection logic to `database/db_access.py`
3. [ ] Move data models to `database/db_models.py`
4. [ ] Extract UI code to `ui/search_interface.py`
5. [ ] Move analysis functions to `analysis/sql_analyzer.py`
6. [ ] Create main script that imports from modules
7. [ ] Add tests for each new module
8. [ ] Deprecate old `exper_sql.py` with migration guide

**Estimated Effort**: 16 hours  
**Impact**: High - Reduces largest complexity hotspot

---

### 2. Consolidate Duplicate Search Modules

**Current State**:
- `exper_sql.py` (1,996 lines)
- `sql_search.py` (1,609 lines)
- `gui_sql.py` (786 lines)
- `conversation_search_gui.py` (460 lines)
- **~30% code duplication across these files**

**Common Patterns**:
- Hardcoded `DB_PATH`
- Similar `PLATFORM_COLORS` dictionaries
- Duplicate connection handling
- Similar query patterns

**Target Architecture**:
```
core/
  ├── __init__.py
  ├── database.py          # Single DB access layer
  ├── search_engine.py     # Unified search logic
  └── config.py            # Centralized configuration

ui/
  ├── __init__.py
  ├── cli_interface.py     # Command-line interface
  └── gui_interface.py     # Graphical interface

# Legacy files moved to deprecated/
deprecated/
  ├── exper_sql.py
  ├── sql_search.py
  └── gui_sql.py
```

**Implementation Steps**:
1. [ ] Create `core/database.py` with unified DB access
2. [ ] Extract common search logic to `core/search_engine.py`
3. [ ] Create `core/config.py` for shared configuration
4. [ ] Refactor `conversation_search_gui.py` to use core modules
5. [ ] Create `ui/cli_interface.py` using core modules
6. [ ] Create `ui/gui_interface.py` using core modules
7. [ ] Add comprehensive tests
8. [ ] Move old files to `deprecated/`
9. [ ] Update documentation

**Estimated Effort**: 24 hours  
**Impact**: High - Eliminates ~2,000 lines of duplication

---

### 3. Extract Configuration to Centralized System

**Current Issues**:
- Hardcoded paths: `"/Users/pup/Desktop/Arch/conversations.db"`
- Magic numbers and hardcoded thresholds
- No environment variable support

**Target Implementation**:
```python
# core/config.py
import os
from pathlib import Path
from typing import Optional

class Config:
    """Centralized configuration management."""
    
    # Database
    DB_PATH: str = os.getenv('DB_PATH', 'conversations.db')
    
    # Directories
    OUTPUT_DIR: Path = Path(os.getenv('OUTPUT_DIR', 'output'))
    VISUALIZATIONS_DIR: Path = Path(os.getenv('VISUALIZATIONS_DIR', 'visualizations'))
    CONTENT_ANALYSIS_DIR: Path = Path(os.getenv('CONTENT_ANALYSIS_DIR', 'content_analysis'))
    LOGS_DIR: Path = Path(os.getenv('LOGS_DIR', 'logs'))
    
    # Pipeline settings
    PIPELINE_CONFIG: str = os.getenv('PIPELINE_CONFIG', 'config/pipeline_config.yaml')
    MAX_RESULTS: int = int(os.getenv('MAX_RESULTS', '100'))
    DEFAULT_PLATFORM: str = os.getenv('DEFAULT_PLATFORM', 'claude')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # UI settings
    PLATFORM_COLORS: dict = {
        'claude': '#FF6B35',
        'chatgpt': '#10A37F',
        'unknown': '#808080'
    }
    
    @classmethod
    def load_from_env(cls, env_file: Optional[str] = '.env') -> 'Config':
        """Load configuration from .env file."""
        if env_file and Path(env_file).exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
        return cls()
```

**Implementation Steps**:
1. [x] Create `.env.example` with all configurable values
2. [ ] Create `core/config.py` with Config class
3. [ ] Update all files to use `Config` instead of hardcoded values
4. [ ] Add python-dotenv to requirements
5. [ ] Update documentation with configuration guide

**Estimated Effort**: 8 hours  
**Impact**: High - Improves portability and flexibility

---

## 🟡 Priority 1: Quality Improvements (Sprint 2)

### 4. Add Type Hints to Pipeline Module

**Current State**: ~10% type hint coverage  
**Target**: >80% type hint coverage in pipeline/

**Implementation Steps**:
1. [ ] Add type hints to `pipeline/pipeline.py`
2. [ ] Add type hints to `pipeline/relevance_scorer.py`
3. [ ] Add type hints to `pipeline/summarizer.py`
4. [ ] Add type hints to `pipeline/duplicate_detector.py`
5. [ ] Add type hints to `pipeline/priority_classifier.py`
6. [ ] Set up mypy configuration
7. [ ] Add mypy check to CI/CD

**Estimated Effort**: 12 hours  
**Impact**: Medium - Improves IDE support and catches bugs

---

### 5. Add Tests for Root Scripts

**Current State**: No tests for root-level utility scripts  
**Target**: >70% coverage for all utility scripts

**Files Needing Tests**:
- [ ] `access_db.py`
- [ ] `analyze_conversations.py`
- [ ] `content_analysis.py`
- [ ] `json_clean.py`
- [ ] `extract_json_samples.py`
- [ ] `extract_samples.py`
- [ ] `uni_parse.py`

**Implementation Steps**:
1. [ ] Create test fixtures for sample data
2. [ ] Add unit tests for each script
3. [ ] Add integration tests for workflows
4. [ ] Set up pytest coverage reporting
5. [ ] Document testing strategy

**Estimated Effort**: 20 hours  
**Impact**: High - Catches regressions and improves confidence

---

### 6. Set Up CI/CD Pipeline

**Goals**:
- Automated testing on every commit
- Code quality checks (linting, formatting)
- Coverage reporting
- Automated deployment (if applicable)

**Implementation Steps**:
1. [ ] Create `.github/workflows/ci.yml`
2. [ ] Add test job (pytest)
3. [ ] Add linting job (flake8)
4. [ ] Add formatting check (black, isort)
5. [ ] Add type checking (mypy)
6. [ ] Add coverage reporting (codecov)
7. [ ] Add status badges to README

**Estimated Effort**: 6 hours  
**Impact**: High - Ensures consistent quality

---

## 🟢 Priority 2: Polish & Optimization (Sprint 3+)

### 7. Improve Documentation

**Tasks**:
- [ ] Add comprehensive docstrings to all public functions
- [ ] Create API reference documentation (Sphinx)
- [ ] Add more usage examples
- [ ] Create troubleshooting guide
- [ ] Add architecture diagrams

**Estimated Effort**: 12 hours

---

### 8. Performance Optimization

**Tasks**:
- [ ] Profile slow operations
- [ ] Optimize database queries
- [ ] Add caching where appropriate
- [ ] Benchmark improvements

**Estimated Effort**: 16 hours

---

### 9. Additional Code Quality Improvements

**Tasks**:
- [ ] Remove unused code
- [ ] Standardize error handling
- [ ] Add logging throughout
- [ ] Create coding standards document

**Estimated Effort**: 8 hours

---

## 📅 Timeline

### Sprint 1 (Weeks 1-2): Critical Refactoring
- Split `exper_sql.py`
- Extract configuration system
- Begin search module consolidation

**Deliverables**:
- New directory structure
- Config system implemented
- 50% reduction in duplication

### Sprint 2 (Weeks 3-4): Quality & Testing
- Complete search module consolidation
- Add type hints to pipeline
- Create tests for utility scripts
- Set up CI/CD

**Deliverables**:
- 80% type hint coverage
- 70% test coverage
- CI/CD pipeline active

### Sprint 3 (Weeks 5-6): Polish & Documentation
- Improve documentation
- Performance optimization
- Code quality improvements

**Deliverables**:
- Complete API documentation
- Performance benchmarks
- Clean, maintainable codebase

---

## 📊 Success Metrics

| Metric | Before | Target | Current |
|--------|--------|--------|---------|
| **Lines of Code** | 8,518 | 7,000 | 8,518 |
| **Duplicate Code** | 30% | <5% | 30% |
| **Test Coverage** | 65% | >80% | 65% |
| **Type Hints** | 10% | >80% | 10% |
| **Largest File** | 1,996 | <600 | 1,996 |
| **Overall Quality** | C+ (68/100) | A- (90/100) | C+ (68/100) |

---

## 🚫 Non-Goals

What we're **NOT** doing:
- Complete rewrite (incremental refactoring only)
- Breaking existing APIs without migration path
- Adding new features during refactoring
- Changing external dependencies unnecessarily

---

## 💡 Notes

### Migration Strategy
- Keep deprecated files for 2 releases
- Provide clear migration guides
- Add deprecation warnings
- Update all examples

### Communication
- Weekly progress updates
- Document all breaking changes
- Seek feedback before major changes
- Keep stakeholders informed

---

## 📚 References

- [CODE_REVIEW.md](CODE_REVIEW.md) - Full code review analysis
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [README.md](README.md) - Project overview

---

**Status Updates**:
- **2026-01-21**: Initial refactoring plan created
- **Next Review**: 2026-02-21 (after Sprint 1)
