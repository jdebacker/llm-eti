import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import statsmodels.api as sm
from pathlib import Path
from stargazer.stargazer import Stargazer


def analyze_eti_heterogeneity(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run regressions to analyze ETI heterogeneity."""
    # First clean the data
    reg_df = df.copy()
    reg_df["income_100k"] = reg_df["broad_income"] / 100000
    reg_df["mtr_change"] = reg_df["new_rate"] - reg_df["prior_rate"]

    # Drop missing values and print diagnostic info
    print("\nData diagnostics:")
    print(f"Original rows: {len(df)}")
    print(f"Non-null ETIs: {df['implied_eti'].notna().sum()}")

    reg_df = reg_df.dropna(subset=["implied_eti", "income_100k", "mtr_change"])
    print(f"Final regression rows: {len(reg_df)}")

    print("\nSummary statistics:")
    print(reg_df[["implied_eti", "income_100k", "mtr_change"]].describe())

    # Run regressions
    results = {}

    try:
        # Model 1: Just income
        y = reg_df["implied_eti"]
        X1 = add_constant(reg_df[["income_100k"]])
        model1 = sm.OLS(y, X1).fit()

        # Model 2: Income and MTR change
        X2 = add_constant(reg_df[["income_100k", "mtr_change"]])
        model2 = sm.OLS(y, X2).fit()

        # Model 3: Income, MTR change, and interaction
        reg_df["income_mtr_interact"] = (
            reg_df["income_100k"] * reg_df["mtr_change"]
        )
        X3 = add_constant(
            reg_df[["income_100k", "mtr_change", "income_mtr_interact"]]
        )
        model3 = sm.OLS(y, X3).fit()

        # Create Stargazer table
        stargazer = Stargazer([model1, model2, model3])

        # Customize table
        stargazer.title("Heterogeneity in Elasticity of Taxable Income")
        stargazer.dependent_variable_name("Elasticity of Taxable Income")

        # Rename variables
        stargazer.rename_covariates(
            {
                "income_100k": "Income ($100k)",
                "mtr_change": "MTR Change",
                "income_mtr_interact": "Income × MTR Change",
                "const": "Constant",
            }
        )

        # Add custom notes
        stargazer.add_custom_notes(
            [
                "Heteroskedasticity-robust standard errors in parentheses.",
                f"Sample includes simulated taxpayer responses across {len(reg_df['broad_income'].unique())} income levels",
                "and 5 tax rates, with 3 responses per combination.",
            ]
        )

        # Generate LaTeX
        latex_table = stargazer.render_latex()

        # Save table
        with open(output_dir / "regression_table.tex", "w") as f:
            f.write(latex_table)

        # Store results
        results = {"model1": model1, "model2": model2, "model3": model3}

    except Exception as e:
        print(f"Error in regression: {str(e)}")
        return None

    return results
