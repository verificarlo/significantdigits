# CLAUDE.md — AI Assistant Guide for `significantdigits`

## Project Overview

`significantdigits` (v0.4.0) is a scientific Python library that computes the number of **significant digits** (or bits) of precision in floating-point computational results. It implements the statistical methodology from the paper [Confidence Intervals for Stochastic Arithmetic](https://arxiv.org/abs/1807.09655).

**Two core metrics:**
- **Significant digits** — bits/digits accurate in a result relative to a reference
- **Contributing digits** — bits/digits that round correctly toward a reference

**Two statistical methods:**
- **CNH** (Centered Normality Hypothesis) — assumes Gaussian-distributed error
- **General** — distribution-free, no normality assumption

---

## Repository Layout

```
significantdigits/
├── src/significantdigits/          # Main package (installed from src/)
│   ├── __init__.py                 # Public API exports
│   ├── _significantdigits.py       # Core implementation (~1374 lines)
│   ├── __main__.py                 # CLI entry point
│   ├── args.py                     # CLI argument parsing
│   ├── stats/                      # Numerical operations abstraction
│   │   ├── dispatch.py             # Routes between dense/sparse/gpu backends
│   │   ├── dense.py                # NumPy-based stats (mean, var, std, errors)
│   │   ├── sparse.py               # SciPy sparse matrix support
│   │   └── gpu.py                  # CuPy GPU backend (optional dependency)
│   └── export/                     # I/O handling
│       ├── stdin.py                # Parse/export Python literals and text
│       └── numpy.py                # .npy binary file support
├── tests/                          # 153 tests across 15 files
│   ├── conftest.py                 # Fixtures, custom CLI options, test helpers
│   ├── utils.py                    # Singleton, Setup for output dirs/paths
│   └── test_*.py                   # Test modules (see Test Suite section)
├── examples/                       # Example scripts and Jupyter notebooks
├── data/                           # Real-world test datasets (.txt)
├── .github/workflows/              # CI/CD pipelines
├── pyproject.toml                  # Project metadata + hatchling build config
├── pytest.ini                      # Pytest configuration and markers
├── setup.py                        # Minimal setup.py wrapper
└── README.md                       # User-facing documentation
```

---

## Development Setup

```bash
# Install in editable mode with all dependencies
pip install -e .

# Or compile requirements from pyproject.toml (matches CI)
pip install pip-tools
python -m piptools compile -o requirements.txt pyproject.toml
pip install . -r requirements.txt
```

**Runtime dependencies:** `numpy>=1.22.0`, `scipy>=1.7.3`, `attrs>=21.2.0`, `icecream>=2.1.3`

**Optional dependencies:** `cupy` for the GPU backend (extras: `gpu`, `gpu-cuda12x`, `gpu-cuda11x`)

**Dev/doc dependencies:** `pytest>=6.2.5`, `pdoc>=14.2.0`, `flake8`

**Python versions supported:** 3.8 – 3.12

---

## Running Tests

```bash
# Run full test suite (verbose, short tracebacks — from pytest.ini defaults)
pytest

# Run only a specific marker
pytest -m performance        # Performance regression tests (may be slow)
pytest -m integration        # CLI, file I/O, end-to-end workflows
pytest -m edge_cases         # Inf/NaN handling, extreme values
pytest -m property_based     # Mathematical invariants, fuzzing
pytest -m validation         # Parameter validation and error handling
pytest -m gpu                # CuPy GPU backend (skipped without CuPy + CUDA device)

# Control sample count for stochastic tests (default: 3)
pytest --nsamples=10

# Run a specific test file or test
pytest tests/test_scalar.py
pytest tests/test_cramer.py::test_significant_digits
```

**Registered pytest markers** (defined in `pytest.ini`):
| Marker | Purpose |
|---|---|
| `performance` | Timing regressions, benchmarks |
| `integration` | CLI + file I/O end-to-end |
| `edge_cases` | Numerical stability, inf/NaN |
| `property_based` | Mathematical property fuzzing |
| `validation` | Input validation, error messages |
| `gpu` | CuPy GPU backend (auto-skipped without CuPy + CUDA device) |

---

## Linting

CI enforces flake8 with these rules:

```bash
# Hard errors (build fails)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Warnings only (exit-zero)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

**Key limits:** max line length = **127**, max cyclomatic complexity = **10**.

---

## Building Documentation

```bash
# Generate HTML docs (from repo root, with pdoc installed)
pdoc -d numpy -t pdoc/template/ --math -o docs significantdigits/
```

Docs are built and deployed to GitHub Pages automatically on push to `main` via `.github/workflows/docs.yml`.

---

## Public API

All public symbols are exported from `significantdigits/__init__.py`:

```python
from significantdigits import (
    significant_digits,       # Compute significant digits
    contributing_digits,      # Compute contributing digits
    change_basis,             # Convert between bases (e.g., bits to digits)
    format_uncertainty,       # Format result with uncertainty notation
    probability_estimation_bernoulli,  # Estimate probability from samples
    minimum_number_of_trials, # Compute minimum required samples
    Method,   # Enum: CNH | General
    Metric,   # Enum: Significant | Contributing
    Error,    # Enum: Absolute | Relative
    InputType, ReferenceType, # Type aliases
    SignificantDigitsException,
)
```

### Key Function Signatures

```python
significant_digits(
    array: InputType,
    reference: ReferenceType | None = None,
    axis: int = 0,
    basis: int = 2,
    error: Error | str,          # "absolute" or "relative"
    method: Method | str,        # "cnh" or "general"
    probability: float = 0.95,
    confidence: float = 0.95,
    shuffle_samples: bool = False,
    dtype: DTypeLike | None = None,
) -> ArrayLike

contributing_digits(...)  # Same signature
```

Enum values accept **case-insensitive strings**: `"cnh"`, `"CNH"`, `"Cnh"` all work.

---

## Code Conventions

### Naming

| Kind | Convention | Example |
|---|---|---|
| Modules/packages | `snake_case` | `_significantdigits.py` |
| Public classes | `PascalCase` | `Method`, `SignificantDigitsException` |
| Public functions | `snake_case` | `significant_digits()` |
| Private functions/classes | `_snake_case` | `_compute_z()`, `_assert_is_numpy_type()` |
| Constants | `UPPER_SNAKE_CASE` | `_VERBOSE_MODE` |
| Type aliases | `PascalCase` | `InputType`, `ReferenceType` |

### Docstrings

Use **NumPy-style** docstrings with reStructuredText math notation:

```python
def my_function(x: np.ndarray, n: int) -> float:
    """
    Short one-line summary.

    Parameters
    ----------
    x : np.ndarray
        Description of x.
    n : int
        Description of n.

    Returns
    -------
    float
        Description of return value.

    Raises
    ------
    TypeError
        When x is not a numpy array.
    """
```

Private helpers use `"""@private"""` as their sole docstring to suppress them from pdoc output.

### Imports

Follow this order (PEP 8):
1. `from __future__ import annotations`
2. Standard library
3. Third-party (`numpy`, `scipy`, `icecream`)
4. Local package imports

### Enum Pattern

Enums extend `AutoName` (value equals the name string):

```python
class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

class Method(AutoName):
    CNH = auto()
    General = auto()
```

Enums support both string and enum-instance comparisons via `.is_cnh()` / `.is_general()` class methods.

### Validation Pattern

Use explicit `_assert_is_*` functions that raise `TypeError` with clear messages:

```python
def _assert_is_numpy_type(x, name):
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(x)}")
```

Do not add defensive checks for impossible states — validate only at public API boundaries.

### Debugging

Verbose output is controlled by the `SD_VERBOSE` environment variable:

```bash
SD_VERBOSE=1 python -m significantdigits ...   # Enable icecream debug output
```

Internal debug calls use `ic(...)` from the `icecream` library. Do not use `print()` for debugging.

---

## CLI Usage

```bash
# Basic usage
significantdigits --array "[1.0, 1.1, 0.9]" --reference 1.0 --method CNH --error absolute

# From stdin
echo "[1.0, 1.1, 0.9]" | significantdigits --method general --error relative

# From .npy files
significantdigits --array data.npy --reference ref.npy --method CNH --error absolute
```

The CLI entry point is `significantdigits.__main__:main`, registered via `pyproject.toml`.

---

## Test Infrastructure

### Key Fixtures (from `tests/conftest.py`)

| Fixture | Type | Description |
|---|---|---|
| `nsamples` | `int` | Number of stochastic samples (CLI: `--nsamples`) |
| `run_fuzzy` | `RunFuzzyTest` | Runs a function N times and collects results |
| `run_significant_digits` | `RunMetricDigitsTest` | Orchestrates significant digits test with CSV output |
| `run_contributing_digits` | `RunMetricDigitsTest` | Same for contributing digits |
| `save` | `Save` | Saves numpy arrays to `.npy` files |
| `load` | `Load` | Loads numpy arrays from `.npy` files |
| `parser` | `ArgumentParser` | Pre-built CLI argument parser |

### Test File Summary

| File | Marker | Focus |
|---|---|---|
| `test_edge_cases.py` | `edge_cases` | Inf, NaN, extreme values |
| `test_validation.py` | `validation` | Parameter errors, type checks |
| `test_property_based.py` | `property_based` | Mathematical invariants |
| `test_integration.py` | `integration` | CLI, I/O, end-to-end |
| `test_performance.py` | `performance` | Timing regressions |
| `test_cramer.py` | — | Real dataset validation |
| `test_parker.py` | — | Parker benchmark |
| `test_higham.py` | — | Higham problem validation |
| `test_scalar.py` | — | Scalar input handling |
| `test_args.py` | — | CLI argument parsing |
| `test_compute_z.py` | — | Internal Z computation |
| `test_scaling_factor.py` | — | Scaling factor computation |
| `test_gpu.py` | `gpu` | CuPy GPU backend and dispatch |
| `test_print_digits.py` | — | Output formatting |

---

## CI/CD Pipelines

| Workflow | Trigger | Actions |
|---|---|---|
| `python-app.yml` | Push/PR to `main` | flake8 lint + pytest on Python 3.8–3.12 |
| `python-publish.yml` | Release published | Build wheel, publish to PyPI |
| `docs.yml` | Push to `main` | pdoc build + deploy to GitHub Pages |

---

## Important Files to Know

- `src/significantdigits/_significantdigits.py` — All core logic; read this first when debugging
- `src/significantdigits/stats/dispatch.py` — Dense vs. sparse routing; touch carefully
- `tests/conftest.py` — Shared fixtures; understand before adding tests
- `tests/utils.py` — `Setup` class manages output directories for test artifacts
- `pyproject.toml` — Single source of truth for versions and dependencies
- `pytest.ini` — Marker definitions and default pytest options

---

## What NOT to Do

- Do not use `print()` for debug output — use `ic()` from `icecream`
- Do not add validation inside private helper functions — validate at public API entry points only
- Do not create `.npy` or `.csv` files in the repo root — they are gitignored and belong in test output directories managed by `Setup`
- Do not exceed 127 characters per line
- Do not break the `AutoName` enum pattern — enum values must equal their name strings
- Do not skip the `--nsamples` fixture when writing stochastic tests — hardcoding sample counts makes tests inflexible
