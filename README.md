# LLM-ETI: What can LLMs tell us about the ETI?

This repository contains code and analysis for the paper "What can LLMs tell us about the ETI?" by Jason DeBacker and Max Ghenis.

## Overview

We investigate how Large Language Models (LLMs) perceive and simulate behavioral responses to tax policy changes, specifically measuring the Elasticity of Taxable Income (ETI). The study includes:

1. **Lab Experiment Replication**: Replicating Pfeil et al. (2024) using LLMs instead of human subjects
2. **Observational Study Replication**: Simulating Gruber & Saez (2002) methodology with LLM responses

## Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API key
- [uv](https://github.com/astral-sh/uv) (auto-installed if missing)

### Installation

```bash
# Install dependencies
make install

# Set your OpenAI API key
export OPENAI_API_KEY=your-key-here
```

### Run Test Analysis

```bash
# Quick test with minimal API calls
make book-test
```

This will:
1. Run test simulations using gpt-4o-mini
2. Generate figures and tables
3. Build the JupyterBook
4. Serve locally at http://localhost:8000

## Project Structure

```
.
├── book/                    # JupyterBook paper
│   ├── _config.yml         # Book configuration
│   ├── intro.md            # Introduction
│   ├── methods.md          # Methodology
│   ├── results/            # Analysis chapters
│   └── scripts/            # Data generation scripts
├── PKNF_2024_replication/  # Lab experiment code
├── ETI_simulations/        # Tax simulation engine
├── results/                # Generated results
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

## JupyterBook Commands

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