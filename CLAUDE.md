# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical CI/CD Workflow Note

**IMPORTANT**: The GitHub Actions workflows are set up with dependencies. The build-book workflow will NOT run if the lint workflow fails. Always check ALL workflows (not just build-book) when monitoring CI:

```bash
# Check ALL recent workflow runs
gh run list --limit=10

# Don't just check build-book workflow - lint failures will block it!
```

## Build and Development Commands

### Installation (uses uv package manager - 10-100x faster than pip)
```bash
make install               # Install deps and create Python 3.13 venv
source .venv/bin/activate  # Activate environment
export OPENAI_API_KEY=xxx  # Required for simulations
```

### Development Workflow
```bash
# ALWAYS format before pushing - lint failures block other CI workflows
make format                # Run black and ruff formatters
make lint                  # Check formatting and run mypy

# Run from project root, not book/ directory
cd book && make test-data  # Generate test data (uses gpt-4o-mini)
cd book && make book       # Build JupyterBook
cd book && make serve      # Serve locally at http://localhost:8000
```

### Testing Single Components
```bash
# Run specific simulation scripts
uv run python book/scripts/run_pknf_simulation.py --test --model gpt-4o-mini
uv run python book/scripts/run_gruber_saez_simulation.py --test --model gpt-4o-mini

# Run regression analysis
uv run python -m llm_eti.simple_regression
```

## Architecture Overview

### Package Structure (`llm_eti/`)
The project is packaged as `llm_eti` with all Python code inside this directory:

- **Core Simulation Engine**: 
  - `simulation_engine.py`: `TaxSimulation` class orchestrates Gruber & Saez style observational simulations
  - `gpt_utils.py`: `GPTClient` wrapper for OpenAI API calls and ETI calculations
  - `config.py`: Configuration management

- **PKNF Lab Experiment Replication** (`PKNF_2024_replication/`):
  - `GPT_PKNF_replication.py`: Main experiment runner simulating lab subjects
  - Implements 16-round tax decision game with different tax schedules

- **Analysis & Visualization**:
  - `analysis.py`: ETI heterogeneity analysis
  - `plotting.py`: Standardized plot generation
  - `regression_utils.py` & `table_utils.py`: Statistical analysis and LaTeX table generation

### JupyterBook Structure (`book/`)
- `_config.yml`: Book configuration (title, author, build settings)
- `_toc.yml`: Table of contents defining chapter order
- `intro.md`, `methods.md`, `results/*.md`: Book content
- `scripts/`: Data generation scripts that import from `llm_eti` package
- `data/`, `figures/`, `tables/`: Generated assets

### Key Design Patterns

1. **Two-Study Approach**: The codebase replicates two different ETI studies:
   - Lab experiment (PKNF 2024): Controlled environment with specific tax schedules
   - Observational study (Gruber & Saez 2002): Simulated real-world tax responses

2. **Simulation Pipeline**: 
   - Scripts in `book/scripts/` orchestrate simulations
   - Results saved to CSV in `book/data/`
   - Figures/tables generated from saved data
   - JupyterBook compiles everything into HTML/PDF

3. **API Cost Management**:
   - `--test` flag limits API calls (10 subjects instead of 100)
   - CI uses `gpt-4o-mini` model to minimize costs
   - Full simulations require explicit commands

## Common Issues and Solutions

### Mypy Errors
The codebase has some type annotation issues that don't affect functionality. If mypy is blocking progress:
- Focus on fixing import errors and syntax warnings first
- Many mypy errors are about missing type annotations (non-critical)

### Import Errors
All imports should use the package form:
```python
from llm_eti import GPTClient, SimulationParams, TaxSimulation
# NOT: from llm_eti.config import SimulationParams
```

### CI Placeholder Data
The CI workflow creates placeholder CSV files to avoid API calls. If plotting fails, ensure test data has sufficient rows and all required columns.