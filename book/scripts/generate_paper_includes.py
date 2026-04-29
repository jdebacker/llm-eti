"""Generate markdown fragments used by the paper build."""

from __future__ import annotations

from pathlib import Path

from llm_eti.paper_results import r

BOOK_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BOOK_DIR / "generated"


def write_fragment(name: str, content: str) -> None:
    path = GENERATED_DIR / name
    path.write_text(f"{content.rstrip()}\n")


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    write_fragment("factorial_design.md", r.table_factorial_design())
    write_fragment("response_distribution.md", r.table_response_dist())
    write_fragment("mean_eti.md", r.table_mean_eti())
    write_fragment("eti_by_income.md", r.table_eti_by_income())

    print(f"Wrote paper fragments to {GENERATED_DIR}")


if __name__ == "__main__":
    main()
