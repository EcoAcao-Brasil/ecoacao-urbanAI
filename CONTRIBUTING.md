# Contributing to UrbanAI

Thank you for your interest in contributing to UrbanAI! This guide will help you get started.

## Code of Conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- GDAL
- Basic understanding of deep learning and remote sensing

### Development Setup

```bash
# Clone repository
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

## Contribution Workflow

### 1. Find or Create an Issue

- Check [existing issues](https://github.com/EcoAcao-Brasil/ecoacao-urbanai/issues)
- Create a new issue describing the problem or feature
- Wait for maintainer feedback before starting major work

### 2. Fork and Branch

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ecoacao-urbanai.git
cd ecoacao-urbanai
git remote add upstream https://github.com/EcoAcao-Brasil/ecoacao-urbanai.git

# Create feature branch
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Write clean, documented code
- Follow the code standards below
- Test your changes locally
- Commit with clear messages

### 4. Submit Pull Request

- Push your branch to your fork
- Open a PR against `main` branch
- Fill out the PR template
- Wait for review

## Code Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these tools:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **Ruff**: Linting
- **mypy**: Type checking

Run formatters:

```bash
black src/
isort src/
ruff check src/
mypy src/
```

These run automatically via pre-commit hooks.

### Code Example

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
        predicted: Predicted values (batch, channels, h, w)
        actual: Actual values (batch, channels, h, w)
        normalize: Whether to normalize residuals

    Returns:
        Tuple of (residuals, statistics)

    Raises:
        ValueError: If shapes don't match

    Example:
        >>> pred = torch.randn(1, 7, 256, 256)
        >>> actual = torch.randn(1, 7, 256, 256)
        >>> residuals, stats = calculate_residuals(pred, actual)
    """
    if predicted.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {actual.shape}")

    residuals = predicted - actual

    if normalize:
        std = torch.std(residuals)
        residuals = residuals / (std + 1e-8)

    stats = {
        "mean_absolute_error": torch.mean(torch.abs(residuals)).item(),
        "rmse": torch.sqrt(torch.mean(residuals**2)).item(),
    }

    return residuals, stats
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ConvLSTMEncoder`)
- **Functions**: `snake_case` (e.g., `calculate_ndvi`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- **Private**: Prefix with `_` (e.g., `_internal_helper`)

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Short description.

    Longer description with details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When this is raised
    """
    pass
```

## Documentation

### Adding Examples

- Add Jupyter notebooks to `examples/notebooks/`
- Add Python scripts to `examples/scripts/`
- Include clear comments and explanations

### Building Documentation

```bash
cd docs
make html
# Open _build/html/index.html
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `chore`: Maintenance

**Examples:**

```
feat(models): add attention mechanism to ConvLSTM

Implement self-attention layer for improved long-range dependencies
in temporal sequences.

Closes #123
```

```
fix(preprocessing): correct NDBI calculation for Landsat 5

The SWIR1 band index was incorrect, causing wrong NDBI values.
Updated band mapping.

Fixes #456
```

## Pull Request Guidelines

### PR Title
Clear, descriptive title following commit message format.

### PR Description
- What changes were made and why
- Link to related issues
- Screenshots/outputs if applicable

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or documented)
- [ ] Runs locally without errors

## Review Process

1. Automated checks must pass (formatting, linting)
2. At least one maintainer approval required
3. All review comments addressed
4. Final approval merges to main

## Project Structure

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

Significant contributors are recognized in:
- GitHub contributors list
- [AUTHORS.md](AUTHORS.md)
- [CHANGELOG.md](CHANGELOG.md)

Thank you for contributing to UrbanAI, by EcoAção Brasil (Brazil EcoAction).
