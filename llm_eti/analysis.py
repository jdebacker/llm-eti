"""
Analysis functions for LLM tax survey results.

This module provides functions for:
- Converting categorical responses to ETI estimates
- Running regressions with controls
- Calculating summary statistics by group
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import pandas as pd
import statsmodels.api as sm

from llm_eti.data_utils import calculate_summary_stats, clean_data, print_diagnostics
from llm_eti.plotting import create_all_plots
from llm_eti.regression_utils import run_model_regressions
from llm_eti.survey import RESPONSE_MIDPOINTS, IncomeResponse
from llm_eti.table_utils import generate_latex_table


def response_to_eti(
    response: IncomeResponse,
    current_rate: float,
    new_rate: float,
) -> Optional[float]:
    """
    Convert a categorical response to an ETI estimate.

    The ETI is calculated as:
        ETI = (% change in income) / (% change in net-of-tax rate)

    Where net-of-tax rate = 1 - marginal_tax_rate

    Args:
        response: Categorical income response
        current_rate: Current marginal tax rate (decimal)
        new_rate: New marginal tax rate (decimal)

    Returns:
        Estimated ETI, or None if rate change is zero
    """
    # Get midpoint estimate for response category
    pct_change_income = RESPONSE_MIDPOINTS[response]

    # Calculate percentage change in net-of-tax rate
    # (1 - new_rate) - (1 - current_rate) / (1 - current_rate)
    # = (current_rate - new_rate) / (1 - current_rate)
    net_of_tax_current = 1 - current_rate
    net_of_tax_new = 1 - new_rate

    if net_of_tax_current == 0:
        return None  # Can't calculate with 100% tax rate

    pct_change_net_of_tax = (net_of_tax_new - net_of_tax_current) / net_of_tax_current

    if abs(pct_change_net_of_tax) < 1e-10:
        return None  # No rate change

    # ETI = % change income / % change net-of-tax
    # Positive ETI means income moves in same direction as net-of-tax rate
    eti = pct_change_income / pct_change_net_of_tax

    return eti


def calculate_mean_eti_by_group(
    data: pd.DataFrame,
    group_col: str,
) -> Dict[str, float]:
    """
    Calculate mean ETI by a grouping variable.

    Args:
        data: DataFrame with 'implied_eti' column
        group_col: Column name to group by

    Returns:
        Dictionary mapping group values to mean ETI
    """
    grouped = data.groupby(group_col)["implied_eti"].mean()
    return cast(Dict[str, float], grouped.to_dict())


def run_eti_regression(
    data: pd.DataFrame,
    dependent_var: str = "implied_eti",
    controls: Optional[List[str]] = None,
) -> Dict[str, Union[Dict, float]]:
    """
    Run OLS regression of ETI on controls.

    Args:
        data: DataFrame with ETI and control variables
        dependent_var: Name of dependent variable column
        controls: List of control variable column names

    Returns:
        Dictionary with:
        - coefficients: Dict mapping variable names to coefficients
        - std_errors: Dict mapping variable names to standard errors
        - r_squared: R-squared value
        - n_obs: Number of observations
    """
    if controls is None:
        controls = []

    # Prepare data
    df = data[[dependent_var] + controls].dropna()

    if len(df) == 0:
        raise ValueError("No valid observations after dropping NAs")

    y = df[dependent_var]
    X = df[controls]
    X = sm.add_constant(X)

    # Run OLS with heteroskedasticity-robust standard errors
    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")

    # Extract results
    coef_names = ["const"] + controls
    coefficients = dict(zip(coef_names, results.params))
    std_errors = dict(zip(coef_names, results.bse))

    return {
        "coefficients": coefficients,
        "std_errors": std_errors,
        "r_squared": results.rsquared,
        "n_obs": len(df),
        "pvalues": dict(zip(coef_names, results.pvalues)),
    }


def analyze_eti_heterogeneity(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run regressions to analyze ETI heterogeneity."""
    # Clean and prepare data
    reg_df = clean_data(df)

    # Calculate summary statistics
    summary_stats = calculate_summary_stats(reg_df)

    # Print diagnostics
    print_diagnostics(df, reg_df, summary_stats)

    results = []

    # Run separate regressions for each model
    for model in reg_df["model"].unique():
        model_df = reg_df[reg_df["model"] == model].copy()
        result = run_model_regressions(model_df, model)
        if result:
            results.append(result)

    if results:
        # Generate combined LaTeX table
        latex_table = generate_latex_table(results, summary_stats)
        with open(output_dir / "regression_table.tex", "w") as f:
            f.write(latex_table)

        # Save summary statistics
        summary_stats.to_csv(output_dir / "summary_stats.csv")

        # Create plots
        create_all_plots(reg_df, output_dir)

    return {"results": results, "summary_stats": summary_stats}
