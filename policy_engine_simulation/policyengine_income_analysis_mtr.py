# %%
import h5py
import pandas as pd
import numpy as np
import os
from policyengine_us import Microsimulation
from huggingface_hub import hf_hub_download

# ── Parameters ─────────────────────────────────────────────────────────────
S           = 1000  # Total sample size across all years
RANDOM_SEED = 42

# Only these two files have the required PolicyEngine-enhanced structure.
# census_cps_2020/2021/2022.h5 exist on HuggingFace but are raw Census files
# that lack household_weight, household_id, and all income variables.
YEARS = {
    2023: "cps_2023.h5",
    2024: "cps_2024.h5",
}

N_PER_YEAR = S // len(YEARS)  # = S/2 for now

# ── Step 1: Download files ─────────────────────────────────────────────────
print("Downloading files")
file_paths = {}
for year, filename in YEARS.items():
    print(f"  {year}: {filename}")
    file_paths[year] = hf_hub_download(
        repo_id="policyengine/policyengine-us-data", filename=filename
    )


# ── Step 2: Process each year ──────────────────────────────────────────────
# For each year:
#   1. Load tax unit IDs and weights from h5
#   2. Use PE's market_income (person-level) aggregated to tax unit for broad income
#   3. Use PE's taxable_income (tax-unit level) directly — no aggregation needed
#   4. Use PE's marginal_tax_rate (person-level) aggregated to tax unit via max
#   5. Expand rows by round(household_weight / 10), minimum 1
#   6. Draw random sample of N_PER_YEAR rows

all_samples = []

for year, filename in YEARS.items():
    print(f"\nProcessing {year}...")
    path = file_paths[year]

    with h5py.File(path, "r") as f:
        household_id     = f["household_id"][:]
        household_weight = f["household_weight"][:]
        person_hh_id     = f["person_household_id"][:]
        person_tu_id     = f["person_tax_unit_id"][:]
        tax_unit_id      = f["tax_unit_id"][:]

    # Instantiate simulation once per year
    baseline = Microsimulation(dataset=path)

    # Broad income: sum person-level market_income to tax unit
    market_income_person = baseline.calculate("market_income", period=year).values
    broad_tu = (
        pd.DataFrame({"tax_unit_id": person_tu_id, "broad_income": market_income_person})
        .groupby("tax_unit_id")["broad_income"]
        .sum()
    )

    # Print diagnostics on market income level before sampling
    print(f"Mean market_income: ${broad_tu.mean():,.2f}")
    print(f"Mean taxable_income: ${pd.Series(taxable_tu).mean():,.2f}")

    # Taxable income: PE's TI is already at the tax unit level — use directly
    taxable_tu = baseline.calculate("taxable_income", period=year).values

    # MTR: PE computes at person level — take max within each tax unit
    mtr_person = baseline.calculate("marginal_tax_rate", period=year).values
    mtr_tu = (
        pd.DataFrame({"tax_unit_id": person_tu_id, "mtr": mtr_person})
        .groupby("tax_unit_id")["mtr"]
        .max()
    )

    # Map household_weight to each tax unit via person bridge
    hh_weight_map = pd.Series(household_weight, index=household_id)
    tu_to_hh = (
        pd.DataFrame({"tax_unit_id": person_tu_id, "household_id": person_hh_id})
        .drop_duplicates("tax_unit_id")
        .set_index("tax_unit_id")["household_id"]
    )

    # Build tax-unit DataFrame
    df = pd.DataFrame({"tax_unit_id": tax_unit_id})
    df["household_weight"] = df["tax_unit_id"].map(tu_to_hh).map(hh_weight_map)
    df["broad_income"]     = df["tax_unit_id"].map(broad_tu)
    df["taxable_income"]   = taxable_tu
    df["mtr"]              = df["tax_unit_id"].map(mtr_tu)
    df["year"]             = year
    df = df[df["household_weight"] > 0].dropna(subset=["household_weight"])
    df = df[df["broad_income"] > 0]

    print(f"  Tax units: {len(df):,}")
    print(f"  Mean broad income:   ${df['broad_income'].mean():,.2f}")
    print(f"  Mean taxable income: ${df['taxable_income'].mean():,.2f}")
    print(f"  Mean MTR:             {df['mtr'].mean():.3f}")

    # Sample N_PER_YEAR rows directly using household_weight as probabilities
    df_year_sample = df.sample(n=N_PER_YEAR, weights="household_weight", replace=True, random_state=RANDOM_SEED).copy()
    print(f"  Sampled: {len(df_year_sample):,}")

    all_samples.append(df_year_sample)

# ── Step 3: Combine all years ──────────────────────────────────────────────
df_final = pd.concat(all_samples, ignore_index=True)

# ── Step 4: Generate R' (post-reform rate) ────────────────────────────────
# R' is a small perturbation around R (MTR), representing a hypothetical
# tax reform. Drawn from N(0, 0.025) and added to MTR.
np.random.seed(RANDOM_SEED)
net_of_tax = 1 - df_final["mtr"]
shock = np.random.lognormal(mean=0, sigma=0.025, size=len(df_final))
net_of_tax_prime = (net_of_tax * shock).clip(0.01, 1.0)
df_final["mtr_prime"] = 1 - net_of_tax_prime

# ── Step 5: Summary statistics ─────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"  Total sample size:    {len(df_final):,}")
print(f"  Years covered:        {sorted(df_final['year'].unique())}")
print(f"{'='*45}")
for yr in sorted(df_final["year"].unique()):
    sub = df_final[df_final["year"] == yr]
    print(f"  {yr} mean broad income:   ${sub['broad_income'].mean():>12,.2f}")
    print(f"  {yr} mean taxable income: ${sub['taxable_income'].mean():>12,.2f}")
    print(f"  {yr} mean MTR:             {sub['mtr'].mean():>12.3f}")
    print(f"  {yr} mean MTR':            {sub['mtr_prime'].mean():>12.3f}")
print(f"{'='*45}")
print(f"  Overall mean broad income:   ${df_final['broad_income'].mean():>12,.2f}")
print(f"  Overall std broad income:    ${df_final['broad_income'].std():>12,.2f}")
print(f"  Overall mean taxable income: ${df_final['taxable_income'].mean():>12,.2f}")
print(f"  Overall std taxable income:  ${df_final['taxable_income'].std():>12,.2f}")
print(f"  Overall mean MTR:             {df_final['mtr'].mean():>12.3f}")
print(f"  Overall mean MTR':            {df_final['mtr_prime'].mean():>12.3f}")
print(f"{'='*45}")

# ── Step 6: Export ─────────────────────────────────────────────────────────
export_cols = ["year", "tax_unit_id", "household_weight", "broad_income", "taxable_income", "mtr", "mtr_prime"]
df_final[export_cols].to_csv("policyengine_sample_incomes.csv", index=False)
print(f"\nExported to: {os.path.abspath('policyengine_sample_incomes.csv')}")
