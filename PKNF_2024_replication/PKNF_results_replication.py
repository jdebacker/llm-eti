# %%
# imports
import pandas as pd
import numpy as np
import linearmodels as lm
import os
import pickle
import plotly.express as px


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
df["post"] = df["round_num"] >= 8
# create labor supply variable
df["labor"] = (df["income"] / 20).astype(int)
df["labor_20"] = df["labor"] <= 20

# %%
# Create Table 5
table5 = (
    df[["treatment", "post", "labor_20"]]
    .groupby(["treatment", "post"])
    .mean()
    .reset_index()
)
# reshape wide so labor_20 is shown for post false and post true
table5 = table5.pivot(
    index="treatment", columns="post", values="labor_20"
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
    df[["treatment", "post", "labor", "max_labor"]]
    .groupby(["treatment", "post", "max_labor"])
    .mean()
    .reset_index()
)
for treat in df_bar["treatment"].unique():
    fig = px.bar(
        df_bar[df_bar["treatment"] == treat],
        x="max_labor",
        y="labor",
        color="post",
        barmode="group",
        title=f"Figure 2: Labor Supply by Treatment ({treat})",
    )
    fig.show()

# %%
# Figure 4
# line plot with mean labor supply by period and treatment
# First, groupby to find mean by treatment, period
df_bar = (
    df[["treatment", "post", "labor", "round_num"]]
    .groupby(["treatment", "post", "round_num"])
    .mean()
    .reset_index()
)
fig = px.line(
    df_bar,
    x="post",
    y="labor",
    color="treatment",
    title="Figure 4: Mean Labor Supply by Period and Treatment",
)
fig.show()

# %%
# Regression results
# labor = constant  + post + treat + post*treat
# Table 6

# %% Use
# Use regression model to estimate the ETI
