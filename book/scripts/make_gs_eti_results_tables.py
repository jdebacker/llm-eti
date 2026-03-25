import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "gruber_saez_results_gpt-4o-mini.csv"
OUTPUT_FILE = DATA_DIR / "gs_eti_regression_results.csv"

print("--------------------------------------------------")
print("Gruber-Saez ETI Regression")
print("--------------------------------------------------")

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} observations from {INPUT_FILE.name}")
model_name = df["model"].iloc[0]
print(f"Model: {model_name}")

# --------------------------------------------------
# 2. Construct log changes
# --------------------------------------------------
df["log_rate_change"] = np.log((1 - df["mtr_prime"]) / (1 - df["mtr"]))
df["log_taxable_income_change"] = np.log(
    df["taxable_income_this"] / df["taxable_income"]
)
df["log_broad_income_change"] = np.log(df["broad_income_this"] / df["broad_income"])

# Drop rows where log change is undefined (zero income or identical pre/post rate)
taxable_sample = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["log_rate_change", "log_taxable_income_change"]
)
broad_sample = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["log_rate_change", "log_broad_income_change"]
)


# --------------------------------------------------
# 3. Run regressions
# --------------------------------------------------
def run_regression(y, X_col, sample, label):
    X = sm.add_constant(sample[X_col])
    y_vals = sample[y]
    reg = sm.OLS(y_vals, X).fit()
    coef = reg.params[X_col]
    pval = reg.pvalues[X_col]
    r2 = reg.rsquared
    nobs = int(reg.nobs)
    mean_eti = sample[label].mean()
    median_eti = sample[label].median()

    def stars(p):
        if p < 0.01:
            return "***"
        elif p < 0.05:
            return "**"
        elif p < 0.1:
            return "*"
        else:
            return ""

    print(
        f"\n  ETI coefficient: {coef:.4f}{stars(pval)}  (p={pval:.4g}, N={nobs}, R²={r2:.4f})"
    )
    print(f"  Mean implied ETI: {mean_eti:.4f} | Median: {median_eti:.4f}")

    return {
        "ETI": f"{coef:.4f}{stars(pval)}",
        "N": nobs,
        "R-squared": round(r2, 4),
        "Mean implied ETI": round(mean_eti, 4),
        "Median implied ETI": round(median_eti, 4),
    }


print("\n[Taxable Income]")
taxable_results = run_regression(
    "log_taxable_income_change",
    "log_rate_change",
    taxable_sample,
    "implied_eti_taxable",
)

print("\n[Broad Income]")
broad_results = run_regression(
    "log_broad_income_change", "log_rate_change", broad_sample, "implied_eti_broad"
)

# --------------------------------------------------
# 4. Save results
# --------------------------------------------------
results_df = pd.DataFrame(
    {
        f"{model_name} (taxable)": taxable_results,
        f"{model_name} (broad)": broad_results,
    }
)

results_df.to_csv(OUTPUT_FILE)
print(f"\n--------------------------------------------------")
print(f"Results saved to {OUTPUT_FILE.name}")
print("--------------------------------------------------")
print(results_df)
