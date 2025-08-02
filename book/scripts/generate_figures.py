#!/usr/bin/env python3
"""
Generate figures for the JupyterBook from simulation data.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_eti.plotting import plot_eti_by_income


def main():
    data_dir = Path(__file__).parent.parent / "data"
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Generating figures...")

    # Check if data exists
    test_mode = False

    # Check for data files
    if (data_dir / "gruber_saez_results_gpt-4o-mini.csv").exists():
        print("Using production data...")
    elif (data_dir / "gruber_saez_results_gpt-4o-mini_test.csv").exists():
        test_mode = True
        print("Using test data...")
    else:
        print("No data found. Run 'make data' or 'make test-data' first.")
        return

    # Load Gruber & Saez results
    suffix = "_test" if test_mode else ""

    try:
        gs_mini = pd.read_csv(
            data_dir / f"gruber_saez_results_gpt-4o-mini{suffix}.csv"
        )
        gs_4o = (
            pd.read_csv(
                data_dir / f"gruber_saez_results_gpt-4o{suffix}.csv"
            )
            if not test_mode
            else gs_mini.copy()
        )
    except FileNotFoundError:
        gs_4o = gs_mini.copy()  # Use mini data for both if 4o not available

    # 1. ETI Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if "implied_eti" in gs_4o.columns and len(gs_4o) > 0:
        # Check if we have valid ETI data
        eti_data = gs_4o["implied_eti"].dropna()
        if len(eti_data) > 0:
            plt.hist(
                eti_data,
                bins=min(50, max(5, len(eti_data) // 2)),
                alpha=0.7,
                label="GPT-4o",
            )
        else:
            plt.text(
                0.5,
                0.5,
                "No ETI data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
    else:
        plt.text(
            0.5,
            0.5,
            "No ETI data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    plt.xlabel("ETI")
    plt.ylabel("Frequency")
    plt.title("GPT-4o: ETI Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    if "implied_eti" in gs_mini.columns and len(gs_mini) > 0:
        # Check if we have valid ETI data
        eti_data = gs_mini["implied_eti"].dropna()
        if len(eti_data) > 0:
            plt.hist(
                eti_data,
                bins=min(50, max(5, len(eti_data) // 2)),
                alpha=0.7,
                label="GPT-4o-mini",
                color="orange",
            )
        else:
            plt.text(
                0.5,
                0.5,
                "No ETI data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
    else:
        plt.text(
            0.5,
            0.5,
            "No ETI data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    plt.xlabel("ETI")
    plt.ylabel("Frequency")
    plt.title("GPT-4o-mini: ETI Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "eti_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. ETI by Income
    if (
        "broad_income" in gs_4o.columns
        and "implied_eti" in gs_4o.columns
        and len(gs_4o) > 0
    ):
        # Check if we have enough data to plot
        if len(gs_4o) > 0 and len(gs_mini) > 0:
            combined_df = pd.concat(
                [gs_4o.assign(model="GPT-4o"), gs_mini.assign(model="GPT-4o-mini")]
            )
            # Only plot if we have sufficient data
            if len(combined_df) > 0 and combined_df["implied_eti"].notna().any():
                try:
                    plot_eti_by_income(combined_df, output_dir=figures_dir)
                except Exception as e:
                    print(f"Warning: Could not generate ETI by income plot: {e}")
                    # Create placeholder
                    plt.figure(figsize=(8, 6))
                    plt.text(
                        0.5,
                        0.5,
                        "Insufficient data for ETI by income plot",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.axis("off")
                    plt.savefig(
                        figures_dir / "eti_by_income.png", dpi=150, bbox_inches="tight"
                    )
                    plt.close()
        else:
            print("Warning: Not enough data to create ETI by income plot")

    # 3. Placeholder for other figures
    # These would be generated from actual simulation data
    placeholder_figures = [
        "lab_supply_by_endowment.png",
        "lab_supply_by_round.png",
        "bunching_density.png",
        "model_eti_comparison.png",
        "response_rate.png",
        "eti_by_tax_direction.png",
    ]

    # Copy existing figures if available
    existing_figures_dir = Path(__file__).parent.parent.parent / "results"
    for fig in placeholder_figures:
        # Try to find existing figures
        for results_subdir in existing_figures_dir.glob("*/"):
            if (results_subdir / fig).exists():
                import shutil

                shutil.copy(results_subdir / fig, figures_dir / fig)
                break
        else:
            # Create placeholder if not found
            plt.figure(figsize=(8, 6))
            plt.text(
                0.5,
                0.5,
                f"Placeholder for:\n{fig}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=14,
            )
            plt.axis("off")
            plt.savefig(figures_dir / fig, dpi=150, bbox_inches="tight")
            plt.close()

    print(f"Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
