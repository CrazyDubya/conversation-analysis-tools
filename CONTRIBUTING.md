# Contributing to Conversation Analysis Tools

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Git

### Setup Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/CrazyDubya/conversation-analysis-tools.git
   cd conversation-analysis-tools
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Production dependencies
   pip install -r requirements.txt
   
   # Development dependencies
   pip install -r requirements-dev.txt
   ```

4. **Set up configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

5. **Install pre-commit hooks** (optional but recommended)
   ```bash
   pre-commit install
   ```

## 📝 Development Guidelines

### Code Style

We follow PEP 8 with some modifications:
- **Line length**: Maximum 100 characters
- **Formatter**: Black with line-length=100
- **Import sorting**: isort with black profile
- **Type hints**: Required for new code in `pipeline/` module

### Running Code Quality Tools

```bash
# Format code
black . --line-length=100

# Sort imports
isort . --profile black --line-length=100

# Lint code
flake8 . --max-line-length=100 --extend-ignore=E203,W503

# Type checking (pipeline module only)
mypy pipeline/ --ignore-missing-imports
```

### Testing

All new features must include tests.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pipeline --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_pipeline.py -v

# Run specific test
pytest tests/test_pipeline.py::test_function_name -v
```

### Test Guidelines
- Place tests in `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include docstrings for complex tests
- Aim for >80% code coverage for new modules

## 🔄 Contribution Workflow

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Branch Naming Conventions
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code patterns
- Add/update tests
- Update documentation
- Keep commits atomic and well-described

### 3. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add relevance scoring feature"
```

#### Commit Message Format
```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat: add duplicate detection threshold configuration

fix: correct relevance score calculation for edge cases

docs: update README with new pipeline examples

refactor: extract database access to separate module
```

### 4. Push and Create Pull Request

```bash
# Push to your branch
git push origin feature/your-feature-name

# Create pull request on GitHub
# - Provide clear description
# - Reference any related issues
# - Ensure all checks pass
```

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (run `black`, `isort`, `flake8`)
- [ ] All tests pass (`pytest tests/`)
- [ ] New code has tests (aim for >80% coverage)
- [ ] Documentation is updated (README, docstrings)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch
- [ ] PR description explains what and why

## 🐛 Reporting Bugs

### Before Submitting
- Check existing issues to avoid duplicates
- Test with the latest version
- Gather relevant information

### Bug Report Template
```markdown
**Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.5]
- Package versions: [from `pip list`]

**Additional Context**
- Error messages
- Stack traces
- Screenshots (if applicable)
```

## 💡 Suggesting Enhancements

### Enhancement Request Template
```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why this feature would be useful.

**Proposed Implementation**
Ideas on how to implement (optional).

**Alternatives Considered**
Other solutions you've considered.
```

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions and classes
- Use Google-style or NumPy-style docstrings
- Include parameter types and return types
- Add usage examples for complex functions

### Docstring Example
```python
def analyze_conversation(conversation_id: int, max_results: int = 100) -> dict:
    """
    Analyze a conversation and return relevance metrics.
    
    Args:
        conversation_id: Unique identifier for the conversation
        max_results: Maximum number of results to return (default: 100)
    
    Returns:
        Dictionary containing analysis results with keys:
        - relevance_score: Float between 0 and 1
        - summary: String summary of the conversation
        - priority: Priority level (CRITICAL, HIGH, MEDIUM, LOW, NONE)
    
    Raises:
        ValueError: If conversation_id is invalid
        DatabaseError: If database connection fails
    
    Example:
        >>> results = analyze_conversation(123, max_results=50)
        >>> print(results['relevance_score'])
        0.87
    """
    # Implementation
```

## 🏗️ Architecture Guidelines

### Adding New Modules
1. Place in appropriate directory (`pipeline/`, `database/`, `ui/`)
2. Create corresponding test file in `tests/`
3. Add to `__init__.py` if it's a public API
4. Update documentation

### Refactoring Existing Code
1. Create issue describing refactoring plan
2. Get feedback before making large changes
3. Maintain backward compatibility when possible
4. Update all affected tests
5. Document breaking changes

## 🎯 Priority Areas

Based on the recent code review, these areas need attention:

### High Priority
1. **Split monolithic files**
   - `exper_sql.py` (1,996 lines)
   - `sql_search.py` (1,609 lines)

2. **Consolidate duplicate code**
   - Search module functionality
   - Database access patterns

3. **Add type hints**
   - Especially in `pipeline/` module

4. **Extract configuration**
   - Move hardcoded paths to config
   - Use environment variables

### Medium Priority
1. Add tests for root-level scripts
2. Improve documentation
3. Set up CI/CD pipeline

## 🤝 Code Review Process

### As a Contributor
- Be open to feedback
- Respond to review comments promptly
- Make requested changes or discuss alternatives
- Keep PR scope focused

### As a Reviewer
- Be respectful and constructive
- Explain the "why" behind suggestions
- Approve when requirements are met
- Help improve the codebase together

## 📞 Getting Help

- **Questions**: Open a GitHub issue with `question` label
- **Discussions**: Use GitHub Discussions for general topics
- **Bugs**: Open issue with detailed description

## 📜 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers this project.

---

Thank you for contributing! Your efforts help make this project better for everyone. 🙌
