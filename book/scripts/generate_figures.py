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

    # Check if data exists - try EDSL format first, then legacy
    test_mode = False
    edsl_mode = False

    # Check for EDSL format first
    if (data_dir / "gruber_saez_results_gpt-4o-mini_edsl.csv").exists():
        edsl_mode = True
        print("Using EDSL format data...")
    elif (data_dir / "gruber_saez_results_gpt-4o-mini_edsl_test.csv").exists():
        edsl_mode = True
        test_mode = True
        print("Using EDSL test data...")
    elif not (data_dir / "gruber_saez_results_gpt-4o-mini.csv").exists():
        if (data_dir / "gruber_saez_results_gpt-4o-mini_test.csv").exists():
            test_mode = True
            print("Using legacy test data...")
        else:
            print("No data found. Run 'make data' or 'make test-data' first.")
            return

    # Load Gruber & Saez results
    suffix = "_test" if test_mode else ""
    edsl_suffix = "_edsl" if edsl_mode else ""

    try:
        gs_mini = pd.read_csv(
            data_dir / f"gruber_saez_results_gpt-4o-mini{edsl_suffix}{suffix}.csv"
        )
        gs_4o = (
            pd.read_csv(
                data_dir / f"gruber_saez_results_gpt-4o{edsl_suffix}{suffix}.csv"
            )
            if not test_mode
            else gs_mini.copy()
        )
    except FileNotFoundError:
        gs_4o = gs_mini.copy()  # Use mini data for both if 4o not available

    # 1. ETI Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if "implied_eti" in gs_4o.columns:
        plt.hist(gs_4o["implied_eti"], bins=50, alpha=0.7, label="GPT-4o")
    plt.xlabel("ETI")
    plt.ylabel("Frequency")
    plt.title("GPT-4o: ETI Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    if "implied_eti" in gs_mini.columns:
        plt.hist(
            gs_mini["implied_eti"],
            bins=50,
            alpha=0.7,
            label="GPT-4o-mini",
            color="orange",
        )
    plt.xlabel("ETI")
    plt.ylabel("Frequency")
    plt.title("GPT-4o-mini: ETI Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "eti_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. ETI by Income
    if "broad_income" in gs_4o.columns and "implied_eti" in gs_4o.columns:
        combined_df = pd.concat(
            [gs_4o.assign(model="GPT-4o"), gs_mini.assign(model="GPT-4o-mini")]
        )

        plot_eti_by_income(combined_df, output_dir=figures_dir)

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
