#%%
import h5py
import shutil
import pandas as pd
import numpy as np
from policyengine_us import Microsimulation

# Parameters 
PERIOD      = 2023   # Tax year
RANDOM_SEED = 42
S = 1000  # number of unique households to sample
N = 5000  # target total weight (controls "effective" population size)

# Load file ───────────────────────────────────────────────────────────
h5_path = r"C:/Users/roxanag/.cache/huggingface/hub/models--policyengine--policyengine-us-data/snapshots/76880421293967143ce0e3ecf880c38e4326024d/pooled_3_year_cps_2023.h5"
h5_path_fixed = h5_path.replace(".h5", "_fixed.h5")

# : Fix immigration status codes ───────────────────────────────────
# The dataset was built with an older version of policyengine_us that used
# different immigration status codes. We remap them to valid current values
# and save to a new file to avoid modifying the original.
shutil.copy2(h5_path, h5_path_fixed)
remap = {
    b"CUBAN_HAITIAN_ENTRANT": b"LEGAL_PERMANENT_RESIDENT",
    b"REFUGEE":               b"LEGAL_PERMANENT_RESIDENT",
    b"DACA":                  b"DACA_TPS",
    b"TPS":                   b"DACA_TPS",
}
with h5py.File(h5_path_fixed, 'r+') as f:
    if 'immigration_status' in f.keys():
        values = f['immigration_status'][:]
        for old, new in remap.items():
            values = np.where(values == old, new, values)
        del f['immigration_status']
        f.create_dataset('immigration_status', data=values)

# Understand the data's hierarchical structure ──────────────────
# The h5 file stores variables at multiple entity levels (person, household,
# tax unit, SPM unit, etc.). Variables at different levels have different row
# counts and cannot be combined into a single DataFrame directly.
# This block prints the distinct sizes so we know which level each variable is at.
with h5py.File(h5_path_fixed, 'r') as f:
    shapes = {key: f[key].shape[0] for key in f.keys()}

unique_sizes = {}
for key, size in shapes.items():
    unique_sizes.setdefault(size, []).append(key)

print("=== Data Hierarchy ===")
for size, keys in sorted(unique_sizes.items()):
    print(f"  {size:>7,} rows: {len(keys)} variables — e.g. {keys[:3]}")

# From the output we know:
#   443,130 = person level
#   233,995 = tax unit level
#   179,514 = SPM unit level
#   172,238 = household level
# Person-level variables link to other levels via bridge IDs:
#   person_household_id, person_tax_unit_id, person_spm_unit_id


#%%
# Load the simulation ────────────────────────────────────────────
# We use the Microsimulation engine only to load the dataset cleanly.
# Note: baseline.calculate() returns zeros for computed variables like AGI
# with this older dataset version, so we construct income variables manually
# from the raw h5 data instead.
baseline = Microsimulation(dataset=h5_path_fixed)

# Load household IDs and weights directly from h5 ────────────────
# We read directly from h5 rather than baseline.calculate() for weights
# because the latter can return zeros depending on dataset/version compatibility.
with h5py.File(h5_path_fixed, 'r') as f:
    household_id     = f['household_id'][:]
    household_weight = f['household_weight'][:]
    person_hh_id     = f['person_household_id'][:]  # bridges person -> household

# Construct broad income at person level, aggregate to household ──
# Broad income = all pre-tax cash income sources.
# We sum at the person level first, then group by household.
broad_income_vars = [
    'employment_income', 'self_employment_income', 'tip_income',
    'social_security_retirement', 'unemployment_compensation',
    'disability_benefits', 'veterans_benefits',
    'taxable_private_pension_income', 'tax_exempt_private_pension_income',
    'long_term_capital_gains', 'short_term_capital_gains',
    'qualified_dividend_income', 'non_qualified_dividend_income',
    'taxable_interest_income', 'tax_exempt_interest_income',
    'rental_income', 'farm_income', 'alimony_income', 'child_support_received'
]

with h5py.File(h5_path_fixed, 'r') as f:
    broad_person = sum(f[v][:] for v in broad_income_vars)

broad_hh = (pd.DataFrame({'household_id': person_hh_id, 'broad_income': broad_person})
              .groupby('household_id')['broad_income'].sum())

# Construct taxable income (AGI proxy) at person level ──────────────────────
# AGI = broad income minus tax-exempt items.
# baseline.calculate("adjusted_gross_income") returns all zeros with this
# dataset/version, so we construct it manually from taxable components only.
taxable_income_vars = [
    'employment_income', 'self_employment_income', 'tip_income',
    'social_security_retirement', 'unemployment_compensation',
    'taxable_private_pension_income', 'taxable_401k_distributions',
    'taxable_ira_distributions', 'long_term_capital_gains',
    'short_term_capital_gains', 'qualified_dividend_income',
    'non_qualified_dividend_income', 'taxable_interest_income',
    'rental_income', 'farm_income', 'alimony_income'
]

with h5py.File(h5_path_fixed, 'r') as f:
    taxable_person = sum(f[v][:] for v in taxable_income_vars)

taxable_hh = (pd.DataFrame({'household_id': person_hh_id, 'taxable_income': taxable_person})
                .groupby('household_id')['taxable_income'].sum())

#%%
# Build household-level DataFrame ────────────────────────────────
# Map aggregated income series back to the household array using household_id
# as the key, ensuring correct alignment across all variables.
df = pd.DataFrame({'household_id': household_id, 'household_weight': household_weight})
df['broad_income']   = df['household_id'].map(broad_hh)
df['taxable_income'] = df['household_id'].map(taxable_hh)

# Drop households with zero or missing weights (not valid for sampling)
df = df[df['household_weight'] > 0].dropna(subset=['household_weight'])
print(f"df shape: {df.shape}")
print(f"weight sum: {df['household_weight'].sum():,.0f}")

# Draw weighted random sample of N households 
# Expand by weights, normalize so rows sum to N
df_sample = df.sample(n=S, weights='household_weight', random_state=RANDOM_SEED).copy()
df_sample['weight_rescaled'] = df_sample['household_weight'] / df_sample['household_weight'].sum() * N

print(f"Rows: {len(df_sample)}")
print(f"Weight sum: {df_sample['weight_rescaled'].sum():,.0f}")

# Compute weighted statistics ────────────────────────────────────
w = df_sample['household_weight'].values
print(f"df_sample shape: {df_sample.shape}")
print(f"df_sample head:\n{df_sample[['household_id','broad_income','taxable_income']].head()}")
#%%
mean_broad   = df_sample['broad_income'].mean()
mean_taxable = df_sample['taxable_income'].mean()
std_broad    = df_sample['broad_income'].std()
std_taxable  = df_sample['taxable_income'].std()

print(f"\n{'='*45}")
print(f"  Sample size:          {N:,} households")
print(f"  Tax year:             {PERIOD}")
print(f"{'='*45}")
print(f"  Mean broad income:    ${mean_broad:>12,.2f}")
print(f"  Std broad income:     ${std_broad:>12,.2f}")
print(f"  Mean taxable income:  ${mean_taxable:>12,.2f}")
print(f"  Std taxable income:   ${std_taxable:>12,.2f}")
print(f"{'='*45}")

export_cols = ['household_id', 'household_weight', 'broad_income', 'taxable_income']
df_sample[export_cols].to_csv("policyengine_sample_incomes.csv", index=False)
print("Exported to policyengine_sample_incomes.csv")
# %%
