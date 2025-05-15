def format_coef(coef, se, pval):
    """Format coefficient with stars and standard error."""
    stars = ""
    if pval < 0.01:
        stars = "^{***}"
    elif pval < 0.05:
        stars = "^{**}"
    elif pval < 0.1:
        stars = "^{*}"
    return f"${coef:.3f}{stars}$ \\\\ ({se:.3f})"


def generate_latex_table(results_dict: dict, summary_stats: dict) -> str:
    """Generate LaTeX table comparing both models."""
    models = ["gpt-4o", "gpt-4o-mini"]

    latex = [
        "\\begin{table}[!htbp] \\centering",
        "  \\caption{Simulated Elasticity of Taxable Income by Model}",
        "\\begin{tabular}{lcc}",
        "\\\\[-1.8ex]\\hline",
        "\\hline \\\\[-1.8ex]",
        " & GPT-4o & GPT-4o-mini \\\\",
        "\\hline \\\\[-1.8ex]",
    ]

    # Panel A: Summary Statistics
    latex.extend(
        [
            "\\multicolumn{3}{l}{\\textbf{Panel A: Summary Statistics}} \\\\",
            "\\\\[-1.8ex]",
        ]
    )

    # Get model results
    model_results = {
        model: next(r for r in results_dict if r["Model"] == model) for model in models
    }

    # Add summary stats
    stats_rows = [
        ("Mean ETI", lambda m: f"${summary_stats.loc[m, 'Mean ETI']:.3f}$"),
        (
            "Median ETI",
            lambda m: f"${summary_stats.loc[m, 'Median ETI']:.3f}$",
        ),
        (
            "Share with no response",
            lambda m: f"{summary_stats.loc[m, 'Share No Response']:.1%}",
        ),
        (
            "Standard deviation",
            lambda m: f"{summary_stats.loc[m, 'Std ETI']:.3f}",
        ),
        (
            "25th percentile ETI",
            lambda m: f"${summary_stats.loc[m, 'P25 ETI']:.3f}$",
        ),
        (
            "75th percentile ETI",
            lambda m: f"${summary_stats.loc[m, 'P75 ETI']:.3f}$",
        ),
        (
            "Number of responses",
            lambda m: f"{int(summary_stats.loc[m, 'N']):,}",
        ),
    ]

    for label, formatter in stats_rows:
        values = [formatter(m) for m in models]
        latex.append(f"{label} & {values[0]} & {values[1]} \\\\")

    # Panel B: Regression Results
    latex.extend(
        [
            "\\\\[-1.8ex]",
            "\\multicolumn{3}{l}{\\textbf{Panel B: Regression Coefficients}} \\\\",
            "\\\\[-1.8ex]",
        ]
    )

    # Add regression results
    var_names = {
        "const": "Constant",
        "income_100k": "Income (\\$100k)",
        "abs_mtr_change": "Absolute MTR Change",
        "income_abs_mtr_interact": "Income $\\times$ Absolute MTR Change",
    }

    # Use model 3 (full specification) for each model
    for var in var_names:
        row = [var_names[var]]
        for model in models:
            reg = model_results[model]["regs"][2]  # Use third regression (full model)
            if var in reg.params.index:
                coef = reg.params[var]
                se = reg.bse[var]
                pval = reg.pvalues[var]
                row.append(format_coef(coef, se, pval))
            else:
                row.append("")
        latex.append(" & ".join(row) + " \\\\")

    # Add R-squared
    r2_values = [model_results[m]["regs"][2].rsquared for m in models]
    latex.append(f"$R^2$ & {r2_values[0]:.3f} & {r2_values[1]:.3f} \\\\")

    # Close table with detailed notes
    latex.extend(
        [
            "\\hline",
            "\\hline \\\\[-1.8ex]",
            "\\multicolumn{3}{p{0.95\\linewidth}}{\\textit{Notes:} ",
            "Heteroskedasticity-robust standard errors in parentheses. ",
            "Panel B reports coefficients from regressions of ETI on income (in \\$100k), marginal tax rate changes (in percentage points), ",
            "and their interaction.} \\\\",
            "\\multicolumn{3}{r}{$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$} \\\\",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(latex)
