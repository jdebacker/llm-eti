# %%
# imports
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import pickle
import plotly.express as px

# get directory of current file
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# set plotly to use white theme
px.defaults.template = "plotly_white"

# %%
# Read in data
DATA_FILENAME = "PKNF_replication_results_gpt-4o-mini"
df = pickle.load(
    open(os.path.join(CUR_DIR, "..", "data", DATA_FILENAME + ".pkl"), "rb")
)

# %%
# Parse model results to get chose income
df["income"] = df["model_answer"].str.extract("(\d+)").astype(float)
# drop if missing income
df = df.dropna(subset=["income"])

# %%
# Create a pre-post variable
df["Post"] = (df["round_num"] >= 8).astype(int)
# create labor supply variable
df["labor"] = (df["income"] / 20).astype(int)
df["labor_20"] = (df["labor"] <= 20).astype(int)
df["lab_supply"] = df["labor"] / df["max_labor"]

# %%
# Create Table 5
table5 = (
    df[["treatment", "Post", "labor_20"]]
    .groupby(["treatment", "Post"])
    .mean()
    .reset_index()
)
# reshape wide so labor_20 is shown for post false and post true
table5 = table5.pivot(
    index="treatment", columns="Post", values="labor_20"
).reset_index()
table5 = table5[["treatment", False, True]]
table5.columns = ["Treatment", "Pre-Reform", "Post-Reform"]
treat_dict = {
    1: "Prog,Prog",
    2: "Prog,Flat25",
    3: "Prog,Flat50",
    4: "Flat25,Prog",
    5: "Flat50,Prog",
}
table5["Treatment"] = table5["Treatment"].map(treat_dict)
# Format decimals

# Save as markdown
table5.to_markdown(os.path.join(CUR_DIR, "tables_figures", "table5.md"), floatfmt=".2f", index=False)
table5.to_latex(os.path.join(CUR_DIR, "tables_figures", "table5.tex"), float_format="%.2f", index=False)
# TODO: add p-value for test of diffs in means

# %%
# Figure 2
# grouped bar plot
# First, groupby to find mean by treatment, post, and max_labor
df_bar = (
    df[["treatment", "Post", "labor", "max_labor"]]
    .groupby(["treatment", "Post", "max_labor"])
    .mean()
    .reset_index()
)
# update treatment to be more descriptive
df_bar["Treatment"] = df_bar["treatment"].map(treat_dict)
treat_num = 0
for treat in df_bar["Treatment"].unique():
    treat_num += 1
    fig = px.histogram(
        df_bar[df_bar["Treatment"] == treat],
        x="max_labor",
        y="labor",
        color="Post",
        barmode="group",
        # title=f"Figure 2: Labor Supply by Treatment ({treat})",
    )
    # Specify the x- and y-axis labels
    fig.update_xaxes(title_text="Maximum Labor")
    fig.update_yaxes(title_text="Labor Supply, in Units")
    fig.write_image(os.path.join(CUR_DIR, "tables_figures", f"LLM_Fig2_{treat_num}.png"))

# %%
# Figure 4
# line plot with mean labor supply by period and treatment
# First, groupby to find mean by treatment, period
df_bar = (
    df[["treatment", "lab_supply", "round_num"]]
    .groupby(["treatment", "round_num"])
    .mean()
    .reset_index()
)
# update treatment to be more descriptive
df_bar["Treatment"] = df_bar["treatment"].map(treat_dict)
fig = px.scatter(
    df_bar,
    x="round_num",
    y="lab_supply",
    color="Treatment",
    # title="Figure 4: Mean Labor Supply by Period and Treatment",
).update_traces(mode='lines+markers')
# Specify the x- and y-axis labels
fig.update_xaxes(title_text="Period")
fig.update_yaxes(title_text="Labor Supply in %")
# put a vertical dashed line at period 8
fig.add_vline(x=7.5, line_dash="dash")
fig.show()
fig.write_image(os.path.join(CUR_DIR, "tables_figures", "LLM_Fig4.png"))

# %%
# Regression results

PKNF_Table6_Col1 = [
    "-0.007", "(0.006)", "-0.052", "(0.022)", "0.083", "(0.009)", "0.947", "(0.055)", "3344", "0.035"
]

# labor = constant  + post + treat + post*treat
# Table 6
# Create a post-treatment interaction variable
df["treated"] = (df["treatment"] > 1).astype(int)
df["post_treat"] = df["Post"] * df["treated"]
df["Treatment"] = df["treatment"].map(treat_dict)
reg_results_dict = {}
for treat in ["Prog,Flat25", "Prog,Flat50", "Flat25,Prog", "Flat50,Prog"]:
    # Keep the treatment and control
    df_reg = df[(df["Treatment"] == treat) | (df["treatment"] == 1)]
    # specify an indicator for the treated group
    model = smf.ols('lab_supply ~ Post + treated + post_treat', data=df_reg)
    # Estimate the model
    results = model.fit()
    reg_results_dict[treat] = [] # initialize list to store results
    reg_results_dict[treat].append(f"{results.params["Post"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["Post"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["treated"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["treated"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["post_treat"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["post_treat"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["Intercept"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["Intercept"]:.3f})")
    # for i, val in enumerate(results.params):
    #     reg_results_dict[treat].append(f"{val:.3f}")
    #     reg_results_dict[treat].append(f"({results.bse[i]:.3f})")
    reg_results_dict[treat].extend([f"{results.nobs:.0f}", f"{results.rsquared:.3f}"])
# put into dataframe
reg_results = pd.DataFrame(reg_results_dict)
reg_results.index = ["Post", "", "Treated", "", "Post*Treated", "", "Constant", "", "N", "R-squared"]
# Save as markdown
reg_results.to_markdown(os.path.join(CUR_DIR, "tables_figures", "table6.md"), index=True)
reg_results.to_latex(os.path.join(CUR_DIR, "tables_figures", "table6.tex"), index=True)

# Make table comparing to PKNF results to LLM results
compare_reg_dict = {
    "PKNF": PKNF_Table6_Col1,
    "LLM": reg_results["Prog,Flat25"]
}
reg_results = pd.DataFrame(compare_reg_dict)
reg_results.index = ["Post", "", "Treated", "", "Post*Treated", "", "Constant", "", "N", "R-squared"]
# Save as markdown
reg_results.to_markdown(os.path.join(CUR_DIR, "tables_figures", "table6_compare.md"), index=True)
reg_results.to_latex(os.path.join(CUR_DIR, "tables_figures", "table6_compare.tex"), index=True)




# %% Use
# Use regression model to estimate the ETI
# do regression as above, but with log income as the dependent variable
# then tax coefficient on the DD and divide by the change in (1-tau)
# e.g. for the Prog,Flat25 group, this is (0.5 - 0.25)