# %%
# imports
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import pickle
import plotly.express as px

# set plotly to use white theme
px.defaults.template = "plotly_white"

# %%
# Read in data
DATA_FILENAME = "PKNF_replication_results"
df = pickle.load(
    open(os.path.join("..", "data", DATA_FILENAME + ".pkl"), "rb")
)

# %%
# Parse model results to get chose income
df["income"] = df["model_answer"].str.extract("(\d+)").astype(float)
# drop if missing income
df = df.dropna(subset=["income"])

# %%
# Create a pre-post variable
df["Post"] = df["round_num"] >= 8
# create labor supply variable
df["labor"] = (df["income"] / 20).astype(int)
df["labor_20"] = df["labor"] <= 20
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
}
table5["Treatment"] = table5["Treatment"].map(treat_dict)
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
for treat in df_bar["Treatment"].unique():
    fig = px.bar(
        df_bar[df_bar["Treatment"] == treat],
        x="max_labor",
        y="labor",
        color="Post",
        barmode="group",
        title=f"Figure 2: Labor Supply by Treatment ({treat})",
    )
    # Specify the x- and y-axis labels
    fig.update_xaxes(title_text="Maximum Labor")
    fig.update_yaxes(title_text="Labor Supply, in Units")
    fig.show()

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
fig = px.line(
    df_bar,
    x="round_num",
    y="lab_supply",
    color="Treatment",
    title="Figure 4: Mean Labor Supply by Period and Treatment",
)
# Specify the x- and y-axis labels
fig.update_xaxes(title_text="Period")
fig.update_yaxes(title_text="Labor Supply in %")
# put a vertical dashed line at period 8
fig.add_vline(x=8, line_dash="dash")
fig.show()

# %%
# Regression results
# labor = constant  + post + treat + post*treat
# Table 6
# Create a post-treatment interaction variable
df["treated"] = df["treatment"] > 1
df["post_treat"] = df["Post"] * df["treated"]
df["Treatment"] = df["treatment"].map(treat_dict)
reg_results_dict = {

}
for treat in df_bar["Treatment"].unique():
    # Keep the treatment and control
    df_reg = df[(df["Treatment"] == treat) | (df["treated"] == 1)]
    # specify an indicator for the treated group
    df_reg["treated_group"] = 0
    df_reg.loc[df_reg["treatment"] > 1, "treated_group"] = 1
    model = smf.ols('lab_supply ~ Post + treated_group + post_treat', data=df_reg)
    # Estimate the model
    results = model.fit()
    print(results.summary())
    # reg_results_dict[treat] = results.params[:]
    # results.bse[:]  # std errors
    # reg_results_dict[treat] = np.extend(reg_results_dict[treat], results.obs, results.rsquared)

# %% Use
# Use regression model to estimate the ETI
