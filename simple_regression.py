# File: simple_regression.py

from data_utils import clean_data
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
import pandas as pd
import numpy as np


def format_coef(coef, se, pvalue):
    """Format coefficient with stars and standard error."""
    stars = ""
    if pvalue < 0.01:
        stars = "$^{***}$"
    elif pvalue < 0.05:
        stars = "$^{**}$"
    elif pvalue < 0.1:
        stars = "$^{*}$"
    return f"{coef:.3f}{stars}"


def run_regressions():
    """Run ETI regressions with different specifications."""
    # Load and clean data
    df = pd.read_csv('results/simulation_4o/raw_responses.csv')
    print(f"\nLoaded GPT-4o simulation data")
    print(f"Number of observations: {len(df):,}")
    
    reg_df = clean_data(df)
    print(f"Observations after cleaning: {len(reg_df):,}")
    
    # Create interaction term
    reg_df['interact'] = reg_df['income_100k'] * reg_df['abs_mtr_change']
    
    # Run four specifications
    models = []
    
    # Model 1: Income only
    X1 = add_constant(reg_df[['income_100k']])
    models.append(OLS(reg_df['implied_eti'], X1).fit(cov_type='HC1'))
    
    # Model 2: MTR change only
    X2 = add_constant(reg_df[['abs_mtr_change']])
    models.append(OLS(reg_df['implied_eti'], X2).fit(cov_type='HC1'))
    
    # Model 3: Both main effects
    X3 = add_constant(reg_df[['income_100k', 'abs_mtr_change']])
    models.append(OLS(reg_df['implied_eti'], X3).fit(cov_type='HC1'))
    
    # Model 4: Full interaction
    X4 = add_constant(reg_df[['income_100k', 'abs_mtr_change', 'interact']])
    models.append(OLS(reg_df['implied_eti'], X4).fit(cov_type='HC1'))
    
    # Calculate statistics for notes
    zero_share = (reg_df['implied_eti'] == 0).mean() * 100
    mean_eti = reg_df['implied_eti'].mean()
    sd_eti = reg_df['implied_eti'].std()
    mean_income = reg_df['income_100k'].mean() * 100
    mean_mtr = reg_df['abs_mtr_change'].mean()
    
    # Create LaTeX table
    latex_table = [
        "\\begin{table}[!htbp]",
        "\\caption{Elasticity of Taxable Income Regression Results}",
        "\\label{tab:eti_reg}",
        "\\begin{center}",
        "\\begin{tabular}{lcccc}",
        "\\hline\\hline",
        "& (1) & (2) & (3) & (4) \\\\",
        "\\hline",
    ]
    
    # Add coefficients and standard errors in aligned blocks
    # First get all coefficients
    coef_lines = {
        'const': ['Constant'],
        'income_100k': ['Income (\\$100k)'],
        'abs_mtr_change': ['$|\\Delta$ MTR$|$'],
        'interact': ['Income $\\times |\\Delta$ MTR$|$']
    }
    
    se_lines = {
        'const': [],
        'income_100k': [],
        'abs_mtr_change': [],
        'interact': []
    }
    
    # Fill in coefficients and standard errors
    for var in coef_lines.keys():
        coef_row = []
        se_row = []
        for model in models:
            if var in model.params:
                coef_row.append(format_coef(
                    model.params[var],
                    model.bse[var],
                    model.pvalues[var]
                ))
                se_row.append(f"({model.bse[var]:.3f})")
            else:
                coef_row.append("")
                se_row.append("")
        coef_lines[var].extend(coef_row)
        se_lines[var].extend(se_row)
    
    # Add coefficient and SE rows to table
    for var in ['const', 'income_100k', 'abs_mtr_change', 'interact']:
        latex_table.append(' & '.join(coef_lines[var]) + ' \\\\')
        if any(se_lines[var]):  # Only add SE row if there are any SEs
            latex_table.append(' & '.join([''] + se_lines[var]) + ' \\\\')
            latex_table.append('\\\\[-8pt]')  # Add some vertical space
    
    # Add model statistics
    latex_table.extend([
        "\\hline",
        f"Observations & {len(reg_df):,} & {len(reg_df):,} & {len(reg_df):,} & {len(reg_df):,} \\\\",
        f"R$^2$ & {models[0].rsquared:.3f} & {models[1].rsquared:.3f} & {models[2].rsquared:.3f} & {models[3].rsquared:.3f} \\\\"
    ])
    
    # Add notes
    latex_table.extend([
        "\\hline\\hline",
        "\\multicolumn{5}{p{0.95\\linewidth}}{\\footnotesize \\textit{Notes:} "
        "Heteroskedasticity-robust standard errors in parentheses. "
        f"Mean ETI is {mean_eti:.3f} (s.d. {sd_eti:.3f}). "
        f"Income is measured in hundreds of thousands of dollars (mean \\${mean_income:.0f},020). "
        f"$|\\Delta$ MTR$|$ is the absolute marginal tax rate change (mean {mean_mtr:.3f}). "
        f"{zero_share:.1f}\\% of responses show zero ETI.} \\\\",
        "\\multicolumn{5}{r}{\\footnotesize $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$} \\\\",
        "\\end{tabular}",
        "\\end{center}",
        "\\end{table}"
    ])
    
    # Save table
    with open('results/simulation_4o/regression_table.tex', 'w') as f:
        f.write('\n'.join(latex_table))
    
    print("\nEnhanced table has been saved to results/simulation_4o/regression_table.tex")
    
    # Print key statistics
    print("\nKey Statistics:")
    print("=" * 80)
    print(f"Share of Zero Responses: {zero_share:.1f}%")
    print(f"Mean ETI: {mean_eti:.3f}")
    print(f"Median ETI: {reg_df['implied_eti'].median():.3f}")
    print(f"Std Dev ETI: {sd_eti:.3f}")
    
    return models


if __name__ == '__main__':
    results = run_regressions()