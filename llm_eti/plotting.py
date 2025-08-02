from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plotting():
    """Set up plotting style."""
    plt.style.use("default")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 12


def plot_eti_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot ETI distribution by model."""
    plt.figure(figsize=(10, 6))
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        # Clip to reasonable range for visualization
        sns.kdeplot(data=model_df["implied_eti"].clip(-2, 2), label=model)
    plt.title("Distribution of ETIs by Model")
    plt.xlabel("ETI")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "eti_distribution.png")
    plt.close()


def plot_eti_by_income(df: pd.DataFrame, output_dir: Path):
    """Plot ETI by income level for each model."""
    plt.figure(figsize=(12, 6))

    # Create income bins
    income_bins = np.arange(50000, 210000, 25000)
    df["income_bin"] = pd.cut(
        df["broad_income"], bins=income_bins, labels=income_bins[:-1]
    )

    # Calculate means by bin and model
    means = (
        df.groupby(["model", "income_bin"], observed=True)["implied_eti"]
        .mean()
        .reset_index()
    )

    # Plot with confidence intervals
    fig, ax = plt.subplots(figsize=(12, 6))

    for model in df["model"].unique():
        model_data = means[means["model"] == model]

        # Add confidence intervals
        model_df = df[df["model"] == model]
        ci_data = []
        for bin_val in income_bins[:-1]:
            bin_data = model_df[model_df["income_bin"] == bin_val]["implied_eti"]
            ci = np.percentile(bin_data, [2.5, 97.5])
            ci_data.append({"bin": bin_val, "lower": ci[0], "upper": ci[1]})
        ci_df = pd.DataFrame(ci_data)

        # Plot mean and CI
        ax.plot(
            model_data["income_bin"],
            model_data["implied_eti"],
            marker="o",
            label=f"{model} (mean)",
        )
        ax.fill_between(ci_df["bin"], ci_df["lower"], ci_df["upper"], alpha=0.2)

    plt.title("ETI by Income Level")
    plt.xlabel("Income")
    plt.ylabel("ETI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(
        income_bins[:-1],
        [f"${x/1000:.0f}k" for x in income_bins[:-1]],
        rotation=45,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "eti_by_income.png")
    plt.close()


def plot_response_patterns(df: pd.DataFrame, output_dir: Path):
    """Plot response patterns."""
    df["mtr_change"] = df["new_rate"] - df["prior_rate"]
    df["any_response"] = df["implied_eti"] != 0

    # Plot 1: Response rate by tax change
    plt.figure(figsize=(12, 6))
    response_rates = (
        df.groupby(["model", "mtr_change"], observed=True)["any_response"]
        .mean()
        .reset_index()
    )

    for model in df["model"].unique():
        model_data = response_rates[response_rates["model"] == model]
        plt.plot(
            model_data["mtr_change"],
            model_data["any_response"],
            marker="o",
            label=model,
        )

    plt.title("Behavioral Response Rate by Tax Rate Change")
    plt.xlabel("Change in Marginal Tax Rate (pp)")
    plt.ylabel("Share Responding")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "response_rate.png")
    plt.close()

    # Plot 2: Mean ETI by tax change direction
    plt.figure(figsize=(12, 6))
    df["tax_change_dir"] = pd.cut(
        df["mtr_change"],
        bins=[-np.inf, -0.001, 0.001, np.inf],
        labels=["Tax Cut", "No Change", "Tax Increase"],
    )

    sns.boxplot(data=df, x="tax_change_dir", y="implied_eti", hue="model")
    plt.title("ETI Distribution by Tax Change Direction")
    plt.xlabel("Tax Rate Change")
    plt.ylabel("ETI")
    plt.tight_layout()
    plt.savefig(output_dir / "eti_by_tax_direction.png")
    plt.close()


def create_all_plots(df: pd.DataFrame, output_dir: Path):
    """Create all plots."""
    setup_plotting()
    plot_eti_distribution(df, output_dir)
    plot_eti_by_income(df, output_dir)
    plot_response_patterns(df, output_dir)
