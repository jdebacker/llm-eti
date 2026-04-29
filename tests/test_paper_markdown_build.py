"""Regression tests for the plain-markdown paper build."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOK_DIR = REPO_ROOT / "book"
CONFIG_PATH = BOOK_DIR / "_config.yml"
INTRO_PATH = BOOK_DIR / "intro.md"
SURVEY_PATH = BOOK_DIR / "results" / "tax_survey.md"
GENERATOR_PATH = BOOK_DIR / "scripts" / "generate_paper_includes.py"
GENERATED_DIR = BOOK_DIR / "generated"
TOC_PATH = BOOK_DIR / "_toc.yml"


def _toc_chapter_paths() -> list[Path]:
    entries = []
    for line in TOC_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("root: "):
            entries.append(stripped.removeprefix("root: ").strip())
        elif stripped.startswith("- file: "):
            entries.append(stripped.removeprefix("- file: ").strip())
    return [BOOK_DIR / f"{entry}.md" for entry in entries]


def _run_generator() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR_PATH)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_published_chapters_use_plain_markdown_sources():
    """Published chapters should not depend on notebook directives."""

    for path in [INTRO_PATH, SURVEY_PATH]:
        content = path.read_text()
        assert "```{code-cell}" not in content
        assert "{eval}`" not in content


def test_book_config_supports_static_build_pipeline():
    """The book config should build only TOC files and enable bibliography support."""

    content = CONFIG_PATH.read_text()

    assert "only_build_toc_files: true" in content
    assert "bibtex_bibfiles:" in content
    assert "- references.bib" in content


def test_results_chapters_are_all_accounted_for_in_toc():
    """Every markdown chapter under book/results should be intentionally published."""

    toc_results = {
        path.name for path in _toc_chapter_paths() if path.parent.name == "results"
    }
    repo_results = {path.name for path in (BOOK_DIR / "results").glob("*.md")}

    assert repo_results == toc_results


def test_generator_writes_expected_markdown_fragments():
    """Paper fragment generation should materialize the markdown includes."""

    _run_generator()

    expected_files = {
        "factorial_design.md": "| Factor | Levels |",
        "response_distribution.md": "| Response | GPT-4o | GPT-4o-mini |",
        "mean_eti.md": "| Scenario | GPT-4o | GPT-4o-mini | Empirical Range |",
        "eti_by_income.md": "| Income | Bracket | GPT-4o ETI | GPT-4o-mini ETI |",
    }

    for filename, marker in expected_files.items():
        path = GENERATED_DIR / filename
        assert path.exists(), f"Missing generated fragment: {path}"
        assert marker in path.read_text()


@pytest.mark.integration
def test_book_build_renders_citations_and_generated_values():
    """A warning-clean build should render bibliography-backed citations and values."""

    _run_generator()

    subprocess.run(
        ["uv", "run", "jupyter-book", "build", "book", "-W"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    intro_html = (BOOK_DIR / "_build" / "html" / "intro.html").read_text()
    references_html = (BOOK_DIR / "_build" / "html" / "references.html").read_text()
    survey_html = (
        BOOK_DIR / "_build" / "html" / "results" / "tax_survey.html"
    ).read_text()

    assert "Horton" in intro_html
    assert "0.36" in intro_html
    assert "80.5%" in intro_html
    assert "Gruber" in references_html
    assert "GPT-4o-mini" in survey_html
