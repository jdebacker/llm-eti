# collect_results.py
from pathlib import Path

import pandas as pd


def read_file(path):
    """Read file content, handling different file types."""
    if path.suffix == ".csv":
        return pd.read_csv(path).head().to_string()
    else:
        with open(path, "r") as f:
            return f.read()


def collect_results(
    results_dir: str = "results/combined_analysis_20241105_150542",
):
    """Collect and format results for sharing."""
    results_path = Path(results_dir)

    output = ["<documents>", "\n# Key Files and Results for ETI Analysis\n"]

    # File structure
    output.append("\n## File Structure")
    for file_path in sorted(results_path.glob("*")):
        output.append(f"- {file_path.name}")

    # Regression table
    reg_table = results_path / "regression_table.tex"
    if reg_table.exists():
        output.extend(
            [
                "\n## Main Regression Table",
                "<document>",
                "<source>regression_table.tex</source>",
                "<document_content>",
                read_file(reg_table),
                "</document_content>",
                "</document>",
            ]
        )

    # Summary stats
    summary_stats = results_path / "summary_stats.csv"
    if summary_stats.exists():
        output.extend(
            [
                "\n## Summary Statistics",
                "<document>",
                "<source>summary_stats.csv</source>",
                "<document_content>",
                read_file(summary_stats),
                "</document_content>",
                "</document>",
            ]
        )

    # Raw data preview
    raw_data = results_path / "combined_results.csv"
    if raw_data.exists():
        output.extend(
            [
                "\n## Raw Data Preview",
                "<document>",
                "<source>combined_results.csv</source>",
                "<document_content>",
                read_file(raw_data),
                "</document_content>",
                "</document>",
            ]
        )

    # List generated plots
    output.append("\n## Generated Plots")
    for plot in results_path.glob("*.png"):
        output.append(f"- {plot.name}")

    output.append("</documents>")

    # Write to file
    with open("results_for_paper.txt", "w") as f:
        f.write("\n".join(output))

    print("Results collected in results_for_paper.txt")


if __name__ == "__main__":
    collect_results()
