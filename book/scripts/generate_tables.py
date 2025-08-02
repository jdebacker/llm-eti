#!/usr/bin/env python3
"""
Generate LaTeX tables for the JupyterBook from simulation data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from table_utils import create_summary_table
from regression_utils import run_regressions


def generate_summary_stats_table(df_4o, df_mini, output_path):
    """Generate summary statistics comparison table."""
    stats = []
    
    for name, df in [('GPT-4o', df_4o), ('GPT-4o-mini', df_mini)]:
        if 'implied_eti' in df.columns:
            stats.append({
                'Model': name,
                'Mean ETI': f"{df['implied_eti'].mean():.3f}",
                'Median ETI': f"{df['implied_eti'].median():.3f}",
                '% Same Income': f"{(df['implied_eti'] == 0).mean() * 100:.1f}%",
                'Observations': f"{len(df):,}"
            })
    
    stats_df = pd.DataFrame(stats)
    
    # Convert to LaTeX
    latex = stats_df.to_latex(index=False, escape=False)
    
    with open(output_path, 'w') as f:
        f.write(latex)


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    tables_dir = Path(__file__).parent.parent / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    print("Generating tables...")
    
    # Check if data exists
    test_mode = False
    if not (data_dir / 'gruber_saez_results_gpt-4o-mini.csv').exists():
        if (data_dir / 'gruber_saez_results_gpt-4o-mini_test.csv').exists():
            test_mode = True
            print("Using test data...")
        else:
            print("No data found. Run 'make data' or 'make test-data' first.")
            return
    
    # Load data
    suffix = '_test' if test_mode else ''
    try:
        gs_mini = pd.read_csv(data_dir / f'gruber_saez_results_gpt-4o-mini{suffix}.csv')
        gs_4o = pd.read_csv(data_dir / f'gruber_saez_results_gpt-4o{suffix}.csv') if not test_mode else gs_mini.copy()
    except FileNotFoundError:
        gs_4o = gs_mini.copy()
    
    # 1. Summary statistics table
    if 'implied_eti' in gs_4o.columns:
        generate_summary_stats_table(gs_4o, gs_mini, tables_dir / 'summary_stats.tex')
    
    # 2. Copy existing regression table if available
    existing_reg_table = Path(__file__).parent.parent.parent / 'results' / 'simulation_4o' / 'regression_table.tex'
    if existing_reg_table.exists():
        import shutil
        shutil.copy(existing_reg_table, tables_dir / 'regression_results.tex')
    else:
        # Create placeholder
        with open(tables_dir / 'regression_results.tex', 'w') as f:
            f.write("% Placeholder for regression results table\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\caption{Regression Results}\n")
            f.write("\\begin{center}\n")
            f.write("\\textit{Run full simulations to generate this table}\n")
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")
    
    print(f"Tables saved to {tables_dir}/")


if __name__ == '__main__':
    main()