from pathlib import Path
import pandas as pd
from llm_eti.data_utils import clean_data, calculate_summary_stats, print_diagnostics
from llm_eti.regression_utils import run_model_regressions
from llm_eti.table_utils import generate_latex_table
from llm_eti.plotting import create_all_plots


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
