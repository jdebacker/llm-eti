import json
from datetime import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis import analyze_eti_heterogeneity


def load_simulation_results(path: Path) -> pd.DataFrame:
    """Load raw responses from a simulation directory."""
    return pd.read_csv(path / "raw_responses.csv")


def plot_model_comparison(combined_df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots between models."""
    plt.style.use("default")

    # ETI by Income Level and Model
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df, x="broad_income", y="implied_eti", hue="model")
    plt.title("ETI Distribution by Income Level and Model")
    plt.xlabel("Broad Income")
    plt.ylabel("Implied ETI")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "eti_by_income_model.png")
    plt.close()

    # ETI by Tax Rate Change and Model
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=combined_df,
        x="mtr_change",
        y="implied_eti",
        hue="model",
        alpha=0.6,
    )
    plt.title("ETI vs Tax Rate Change by Model")
    plt.xlabel("Change in Marginal Tax Rate")
    plt.ylabel("Implied ETI")
    plt.tight_layout()
    plt.savefig(output_dir / "eti_by_mtr_model.png")
    plt.close()


def save_summary_stats(df: pd.DataFrame, output_dir: Path):
    """Save summary statistics by model."""
    summary_dict = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        summary_dict[model] = {
            "eti_stats": {
                "count": int(model_df["implied_eti"].count()),
                "mean": float(model_df["implied_eti"].mean()),
                "std": float(model_df["implied_eti"].std()),
                "min": float(model_df["implied_eti"].min()),
                "max": float(model_df["implied_eti"].max()),
            },
            "income_stats": {
                "mean": float(model_df["parsed_income"].mean()),
                "std": float(model_df["parsed_income"].std()),
            },
        }

    with open(output_dir / "model_comparison_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)


def print_summary_stats(df: pd.DataFrame):
    """Print key summary statistics."""
    print("\nKey Findings:")
    print("-------------")
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n{model}:")
        print(f"Sample size: {len(model_df):,}")
        print(f"Average ETI: {model_df['implied_eti'].mean():.3f}")
        print(f"Standard Deviation: {model_df['implied_eti'].std():.3f}")
        print(f"Median ETI: {model_df['implied_eti'].median():.3f}")
        print(
            f"Income range: ${model_df['broad_income'].min():,.0f} - ${model_df['broad_income'].max():,.0f}"
        )


@click.command()
@click.argument("mini_path", type=click.Path(exists=True))
@click.argument("full_path", type=click.Path(exists=True))
@click.option(
    "--output",
    default=None,
    help="Output directory name (defaults to timestamp)",
)
def combine_and_analyze(mini_path, full_path, output):
    """Combine and analyze results from GPT-4o-mini and GPT-4o simulations."""

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / (output or f"combined_analysis_{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load both sets of results
    mini_df = load_simulation_results(Path(mini_path))
    full_df = load_simulation_results(Path(full_path))

    print("\nLoaded data:")
    print(f"GPT-4o-mini: {len(mini_df):,} observations")
    print(f"GPT-4o: {len(full_df):,} observations")

    # Add model indicators
    mini_df["model"] = "gpt-4o-mini"
    full_df["model"] = "gpt-4o"

    # Combine results
    combined_df = pd.concat([mini_df, full_df], ignore_index=True)

    # Calculate MTR change for visualization
    combined_df["mtr_change"] = combined_df["new_rate"] - combined_df["prior_rate"]
    combined_df["abs_mtr_change"] = np.abs(combined_df["mtr_change"])

    # Remove extreme outliers for visualization
    viz_df = combined_df[
        (combined_df["implied_eti"] >= combined_df["implied_eti"].quantile(0.01))
        & (combined_df["implied_eti"] <= combined_df["implied_eti"].quantile(0.99))
    ].copy()

    # Save combined dataset
    combined_df.to_csv(output_dir / "combined_results.csv", index=False)
    print(f"\nCombined results saved to {output_dir}/combined_results.csv")

    # Generate comparison plots
    plot_model_comparison(viz_df, output_dir)
    print(f"Comparison plots saved to {output_dir}")

    # Save summary statistics
    save_summary_stats(combined_df, output_dir)
    print(f"Summary statistics saved to {output_dir}/model_comparison_summary.json")

    # Run regression analysis
    _ = analyze_eti_heterogeneity(combined_df, output_dir)
    print(f"Regression results saved to {output_dir}/regression_table.tex")

    # Print summary statistics
    print_summary_stats(combined_df)


if __name__ == "__main__":
    combine_and_analyze()
