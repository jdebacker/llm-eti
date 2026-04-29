# LLM-ETI: What can LLMs tell us about the ETI?

This repository contains code and analysis for the paper "What can LLMs tell us about the ETI?" by Jason DeBacker and Max Ghenis.

## Overview

We investigate how Large Language Models (LLMs) perceive and simulate behavioral responses to tax policy changes, specifically measuring the Elasticity of Taxable Income (ETI).

The current paper includes:

1. **Lab Experiment Replication**: Replicating Pfeil et al. (2024) using LLMs instead of human subjects
2. **Tax Response Survey**: A factorial survey of taxpayer personas and tax shocks

Exploratory writeups that are not part of the published book live outside the book TOC.

## Quick Start

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv)
- Expected Parrot API key (for EDSL)

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make install

# Set your Expected Parrot API key
export EXPECTED_PARROT_API_KEY=your-key-here
```

### For Collaborators

To ensure everyone uses the same environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MaxGhenis/llm-eti.git
   cd llm-eti
   ```

2. **Install dependencies**:
   ```bash
   make install
   source .venv/bin/activate
   ```

3. **Verify installation**:
   ```bash
   python --version  # Should show 3.12.x inside .venv
   uv --version      # Should show uv version
   ```

### Run Test Analysis

```bash
# Quick test with minimal API calls
make book-test
```

This will:
1. Run test simulations using gpt-4o-mini
2. Generate figures, tables, and markdown include fragments
3. Build the Jupyter Book
4. Serve locally at http://localhost:8000

## Project Structure

```
.
├── book/                    # Published Jupyter Book paper
│   ├── _config.yml         # Book configuration
│   ├── results/            # Published results chapters
│   ├── drafts/             # Unpublished exploratory chapters
│   ├── generated/          # Generated markdown fragments
│   └── scripts/            # Data generation scripts
├── llm_eti/                # Python package and analysis code
├── results/                # Generated simulation outputs
├── tests/                  # Regression and integration tests
└── pyproject.toml          # Project dependencies
```

## Full Analysis Pipeline

To run the complete analysis (warning: expensive API calls!):

```bash
# Run full simulations
make run-simulation-4o  # GPT-4o
make run-simulation     # GPT-4o-mini

# Build the book
cd book
make all
```

## Jupyter Book Commands

- `make book` - Build the JupyterBook
- `make book-serve` - Serve locally
- `make book-pdf` - Generate PDF
- `make book-test` - Run test pipeline

## Development

```bash
# Format code
cd book && make format

# Run linters
cd book && make lint

# Run tests
cd book && make test
```

## Citation

```bibtex
@article{debacker2025llmeti,
  title={What can LLMs tell us about the ETI?},
  author={DeBacker, Jason and Ghenis, Max},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
