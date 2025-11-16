# Contributing to FluxPipeline

First off, thank you for considering contributing to FluxPipeline! It's people like you that make FluxPipeline such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by the [FluxPipeline Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Git
- Python 3.10 or higher (3.11-3.13 recommended)
- CUDA toolkit (if using NVIDIA GPU)
- Docker (optional, for containerized development)

### First Contribution

Unsure where to begin? You can start by looking through these issues:
- **Beginner issues** - issues labeled `good first issue`
- **Help wanted issues** - issues labeled `help wanted`

## Development Setup

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/flux_pipeline.git
   cd flux_pipeline
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/Xza85hrf/flux_pipeline.git
   ```

4. **Set up your environment**
   ```bash
   # Copy environment template
   cp .env.example .env

   # Create conda environment (recommended)
   conda create -n flux-dev python=3.12 -y
   conda activate flux-dev

   # Or use venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

5. **Install in development mode**
   ```bash
   # Install with dev dependencies
   pip install -e ".[dev]"

   # For CUDA support
   pip install -e ".[dev,cuda]"
   ```

6. **Install pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

7. **Verify installation**
   ```bash
   # Run tests
   pytest

   # Run linting
   ruff check .
   black --check .
   mypy .
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or log output.

**Environment:**
 - OS: [e.g., Ubuntu 22.04]
 - Python version: [e.g., 3.12]
 - CUDA version: [e.g., 12.4]
 - GPU: [e.g., RTX 4090]

**Additional context**
Any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

1. Use a clear and descriptive title
2. Provide a detailed description of the suggested enhancement
3. Explain why this enhancement would be useful
4. List any alternative solutions you've considered

### Your First Code Contribution

#### Local Development

1. **Create a branch** for your work
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. **Make your changes** following our style guidelines

3. **Add tests** for new functionality

4. **Run the test suite**
   ```bash
   pytest tests/ -v
   ```

5. **Lint your code**
   ```bash
   ruff check .
   black .
   mypy .
   ```

## Development Workflow

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes
- `chore/` - Maintenance tasks

Examples:
- `feature/add-style-transfer`
- `fix/memory-leak-gpu-cleanup`
- `docs/update-installation-guide`

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream changes into your main branch
git checkout main
git merge upstream/main

# Rebase your feature branch
git checkout feature/your-feature
git rebase main
```

## Code Style Guidelines

### Python Style

We follow **PEP 8** with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted using isort/Ruff
- **Type hints**: Encouraged but not required
- **Docstrings**: Required for public functions/classes (Google style)

### Formatting Tools

- **Black**: Code formatting
- **Ruff**: Fast linting and import sorting
- **MyPy**: Static type checking

```bash
# Auto-format code
black .

# Fix auto-fixable linting issues
ruff check --fix .

# Type check
mypy .
```

### Docstring Style

Use Google-style docstrings:

```python
def generate_image(prompt: str, steps: int = 4) -> Image:
    """Generate an image from a text prompt.

    Args:
        prompt: The text description of the image to generate.
        steps: Number of inference steps (1-4 recommended).

    Returns:
        Generated PIL Image object.

    Raises:
        ValueError: If steps is outside valid range.
        RuntimeError: If model loading fails.

    Example:
        >>> image = generate_image("A sunset over mountains", steps=4)
        >>> image.save("output.png")
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage (target: 80%+)
- Use pytest fixtures for setup/teardown
- Mock external dependencies (GPU, network, etc.)

### Test Structure

```python
import pytest
from your_module import YourClass

class TestYourClass:
    """Tests for YourClass."""

    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return YourClass()

    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        result = instance.method()
        assert result == expected_value

    @pytest.mark.gpu
    def test_gpu_functionality(self, instance):
        """Test GPU-specific functionality."""
        # This test only runs when GPU is available
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_pipeline.py

# Run tests matching pattern
pytest -k "test_memory"

# Run only unit tests
pytest -m unit

# Skip GPU tests
pytest -m "not gpu"
```

## Commit Message Guidelines

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Test additions/changes
- **chore**: Maintenance tasks
- **perf**: Performance improvements

### Examples

```
feat(pipeline): add batch processing support

Implement batch processing functionality to generate multiple images
in a single pass, improving throughput by 3x.

Closes #123
```

```
fix(memory): resolve GPU memory leak in cleanup

Fixed memory leak that occurred when models were unloaded.
Added proper context manager for GPU resources.

Fixes #456
```

### Best Practices

- Use present tense ("add feature" not "added feature")
- Use imperative mood ("move cursor" not "moves cursor")
- Limit first line to 72 characters
- Reference issues/PRs in footer

## Pull Request Process

### Before Submitting

1. **Self-review** your changes
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Run the full test suite**
5. **Ensure all CI checks pass**
6. **Update CHANGELOG.md** with your changes

### Submitting a PR

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

2. **Create Pull Request** on GitHub
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Request reviewers if needed

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Tests added/updated
   - [ ] All tests pass locally
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated

   ## Related Issues
   Closes #XXX
   ```

4. **Respond to feedback**
   - Address review comments promptly
   - Push additional commits to the same branch
   - Request re-review when ready

### After Merge

1. **Delete your branch** (optional)
   ```bash
   git branch -d feature/your-feature
   git push origin --delete feature/your-feature
   ```

2. **Update your local main**
   ```bash
   git checkout main
   git pull upstream main
   ```

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check README and docs/ folder

### Recognition

Contributors are recognized in:
- CHANGELOG.md for their contributions
- GitHub contributors page
- Release notes (for significant contributions)

## Development Tips

### Performance Profiling

```python
# Use cProfile for performance analysis
python -m cProfile -o profile.stats main.py

# Use memory_profiler
from memory_profiler import profile

@profile
def your_function():
    pass
```

### Debugging

```python
# Use pdb for debugging
import pdb; pdb.set_trace()

# Or use breakpoint() (Python 3.7+)
breakpoint()
```

### GPU Debugging

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Questions?

Don't hesitate to ask! Create an issue with the `question` label, and we'll be happy to help.

---

**Thank you for contributing to FluxPipeline!** 🎉
