# Contributing to UrbanAI

Thank you for your interest in contributing to UrbanAI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- GDAL (for geospatial operations)
- Basic understanding of deep learning and remote sensing

### Development Environment

We recommend using a virtual environment:

```bash
# Clone the repository
git clone https://github.com/EcoAcao-Brasil/ecoacao-urbanai.git
cd ecoacao-urbanai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development Setup

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev python3-dev
```

#### macOS
```bash
brew install gdal
```

#### Windows
Download and install GDAL from [GISInternals](https://www.gisinternals.com/)

### Python Dependencies

All Python dependencies are managed in `pyproject.toml`:

```bash
# Core dependencies
pip install -e .

# Development dependencies
pip install -e ".[dev]"

# Documentation dependencies
pip install -e ".[docs]"
```

## Contribution Workflow

### 1. Find or Create an Issue

- Check existing [issues](https://github.com/EcoAcao-Brasil/ecoacao-urbanai/issues)
- Create a new issue if needed, describing the problem or feature
- Wait for maintainer feedback before starting major work

### 2. Fork and Branch

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ecoacao-urbanai.git
cd ecoacao-urbanai
git remote add upstream https://github.com/EcoAcao-Brasil/ecoacao-urbanai.git

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Write code following our [code standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed
- Commit with clear, descriptive messages

### 4. Test Locally

```bash
# Run linters
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/

# Run tests
pytest tests/ -v --cov=urbanai

# Build docs
cd docs && make html
```

### 5. Submit Pull Request

- Push your branch to your fork
- Open a PR against `main` branch
- Fill out the PR template completely
- Wait for review and address feedback

## Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 100 characters (enforced by Black)
- **Imports**: Sorted with isort
- **Type Hints**: Required for all functions
- **Docstrings**: Google style format

### Code Formatting

We use automated formatters:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

These run automatically via pre-commit hooks.

### Example Function

```python
from typing import Optional, Tuple

import numpy as np
import torch


def calculate_residuals(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate residuals between predicted and actual values.

    Args:
        predicted: Predicted values tensor of shape (batch, channels, h, w)
        actual: Actual values tensor of shape (batch, channels, h, w)
        normalize: Whether to normalize residuals by standard deviation

    Returns:
        Tuple of (residuals tensor, statistics dictionary)

    Raises:
        ValueError: If tensor shapes don't match

    Example:
        >>> predicted = torch.randn(1, 7, 256, 256)
        >>> actual = torch.randn(1, 7, 256, 256)
        >>> residuals, stats = calculate_residuals(predicted, actual)
        >>> print(stats['mean_absolute_error'])
        0.123
    """
    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs actual {actual.shape}"
        )

    residuals = predicted - actual

    if normalize:
        std = torch.std(residuals)
        residuals = residuals / (std + 1e-8)

    stats = {
        "mean_absolute_error": torch.mean(torch.abs(residuals)).item(),
        "root_mean_square_error": torch.sqrt(torch.mean(residuals**2)).item(),
    }

    return residuals, stats
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ConvLSTMEncoder`)
- **Functions**: `snake_case` (e.g., `calculate_ndvi`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- **Private**: Prefix with `_` (e.g., `_internal_helper`)

## Testing Guidelines

### Test Structure

```
tests/
├── test_preprocessing/
│   ├── test_raster_loader.py
│   ├── test_band_processor.py
│   └── test_tocantins_integration.py
├── test_models/
│   ├── test_convlstm.py
│   └── test_encoder_decoder.py
├── test_training/
│   └── test_trainer.py
└── integration/
    └── test_end_to_end.py
```

### Writing Tests

```python
import pytest
import torch
from urbanai.models import ConvLSTM


class TestConvLSTM:
    """Test suite for ConvLSTM model."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return ConvLSTM(
            input_dim=7,
            hidden_dims=[64, 128],
            kernel_size=(3, 3),
            num_layers=2,
        )

    def test_forward_pass(self, model):
        """Test forward pass with valid input."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 7, 64, 64)

        outputs, states = model(x)

        assert len(outputs) == 1  # Single output layer
        assert outputs[0].shape == (batch_size, seq_len, 128, 64, 64)

    def test_invalid_input_shape(self, model):
        """Test handling of invalid input."""
        x = torch.randn(2, 10, 5, 64, 64)  # Wrong channel count

        with pytest.raises(RuntimeError):
            model(x)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, model, batch_size):
        """Test model with different batch sizes."""
        x = torch.randn(batch_size, 10, 7, 64, 64)
        outputs, _ = model(x)
        assert outputs[0].shape[0] == batch_size
```

### Test Coverage

- Aim for **>80% code coverage**
- All public APIs must be tested
- Include edge cases and error conditions
- Use parametrized tests for multiple scenarios

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models/test_convlstm.py

# Run with coverage
pytest tests/ --cov=urbanai --cov-report=html

# Run specific test
pytest tests/test_models/test_convlstm.py::TestConvLSTM::test_forward_pass
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Short one-line description.

    Longer description with more details about what the function does,
    how it works, and any important notes.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When and why this exception is raised
        TypeError: When and why this exception is raised

    Example:
        >>> result = function_name(42, "test")
        >>> print(result)
        True

    Note:
        Additional notes or warnings about the function.
    """
    pass
```

### Building Documentation

```bash
cd docs
make html
# Open _build/html/index.html
```

### Adding Examples

- Add Jupyter notebooks to `examples/notebooks/`
- Add Python scripts to `examples/scripts/`
- Include clear comments and explanations
- Test all examples before committing

## Submitting Changes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(models): add transformer-based encoder option

Implement transformer encoder as alternative to ConvLSTM encoder
for handling long temporal sequences more efficiently.

Closes #123
```

```
fix(preprocessing): correct NDBI calculation for Landsat 5

The SWIR1 band index was incorrect for Landsat 5, causing
wrong NDBI values. Updated band mapping.

Fixes #456
```

### Pull Request Guidelines

1. **Title**: Clear, descriptive title following commit message format
2. **Description**: 
   - What changes were made and why
   - Link to related issues
   - Screenshots/outputs if applicable
3. **Checklist**:
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] No breaking changes (or documented)

### Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer approval required
3. All review comments must be addressed
4. Final approval merges to main

## Project Structure

Understanding the codebase:

```
src/urbanai/
├── preprocessing/    # Data loading and feature calculation
├── models/          # Neural network architectures
├── training/        # Training orchestration
├── prediction/      # Inference and forecasting
├── analysis/        # Post-prediction analysis
├── visualization/   # Plotting and mapping
├── io/             # File I/O operations
├── utils/          # Utility functions
└── pipeline.py     # Main pipeline orchestrator
```

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/EcoAcao-Brasil/ecoacao-urbanai/discussions)
- **Bugs**: Open an [Issue](https://github.com/EcoAcao-Brasil/ecoacao-urbanai/issues)
- **Email**: isaque@ecoacaobrasil.org

## Recognition

Contributors are recognized in:
- GitHub contributors list
- CHANGELOG.md for significant contributions
- AUTHORS.md file

Thank you for contributing to UrbanAI! Together we're building tools for a more sustainable urban future. 🌍
