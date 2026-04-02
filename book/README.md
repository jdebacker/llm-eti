# LLM-ETI Jupyter Book

This directory contains the published book for "What can LLMs tell us about the ETI?"

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (will be auto-installed if missing)

## Quick Start

```bash
# Install dependencies
make install

# Build and serve the book locally
make serve
```

That's it! The book will be available at http://localhost:8000

The build pipeline now uses plain MyST markdown plus generated include files. The
published book only includes chapters listed in `_toc.yml`; exploratory material
lives outside `results/`.

## Full Pipeline

To run the complete analysis with LLM simulations:

```bash
# Set your Expected Parrot API key
export EXPECTED_PARROT_API_KEY=your-key-here

# Run test simulations (uses gpt-4o-mini, cheaper)
make test-data

# Generate figures, tables, and markdown includes
make figures tables generate-paper

# Build everything
make all
```

For full simulations (warning: expensive API calls!):

```bash
# Run full simulations with multiple models
make data
```

## Structure

- `intro.md`, `methods.md`, `discussion.md`, `references.md` - Published core chapters
- `results/` - Published results chapters listed in `_toc.yml`
- `drafts/` - Unpublished exploratory writeups kept out of the book TOC
- `generated/` - Generated markdown fragments included by published chapters
- `scripts/` - Python scripts for simulations and build-time content generation
- `data/`, `figures/`, `tables/` - Generated analysis artifacts

## CI/CD

The GitHub Actions workflow builds the book from generated artifacts and
published chapters, then deploys it from `main`.

## Customization

Edit `_config.yml` to change book metadata, `_toc.yml` to modify the published
table of contents, and `scripts/generate_paper_includes.py` if a chapter needs
new generated markdown.
