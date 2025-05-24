import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for analysis."""
    reg_df = df.copy()

    # Convert columns to numeric and clean
    reg_df["income_100k"] = (
        pd.to_numeric(reg_df["broad_income"], errors="coerce") / 100000
    )
    reg_df["mtr_change"] = pd.to_numeric(
        reg_df["new_rate"], errors="coerce"
    ) - pd.to_numeric(reg_df["prior_rate"], errors="coerce")
    reg_df["implied_eti"] = pd.to_numeric(
        reg_df["implied_eti"], errors="coerce"
    )
    reg_df["abs_mtr_change"] = np.abs(reg_df.mtr_change)

    # Drop any invalid values
    reg_df = reg_df.dropna(subset=["income_100k", "mtr_change", "implied_eti"])

    return reg_df


def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics by model."""
    summary_stats = (
        df.groupby("model")
        .agg(
            {
                "implied_eti": [
                    "count",
                    "mean",
                    "median",
                    "std",
                    lambda x: (x == 0).mean(),
                    lambda x: np.percentile(x, 25),
                    lambda x: np.percentile(x, 75),
                ]
            }
        )
        .round(4)
    )
    summary_stats.columns = [
        "N",
        "Mean ETI",
        "Median ETI",
        "Std ETI",
        "Share No Response",
        "P25 ETI",
        "P75 ETI",
    ]
    return summary_stats


def print_diagnostics(
    original_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    summary_stats: pd.DataFrame,
):
    """Print data diagnostics."""
    print("\nData diagnostics:")
    print(f"Original rows: {len(original_df)}")
    print(f"Non-null ETIs: {original_df['implied_eti'].notna().sum()}")
    print(f"Clean rows after conversion: {len(clean_df)}")

    print("\nSummary by model:")
    print(summary_stats)
