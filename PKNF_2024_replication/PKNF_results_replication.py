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
df = pickle.load(open(os.path.join("..", "data", DATA_FILENAME + ".pkl"), "rb"))

# %%
# Parse model results to get chose income
df["income"] = df["model_answer"].str.extract(r"Chose income: (\d+)")

# %%
# Create a pre-post variable
df['post'] = df['round_num'] >= 8
# create labor supply variable
df["labor"] = df["income"].astype(int) / 20
df["labor_20"] = df["labor"] <= 20

# %%
# Create Table 5
table5 = df.groupby(["treatment", "post"]).mean("labor_20").reset_index()
# TODO: add p-value for test of diffs in means

# %%
# Figure 2
# grouped bar plot
for treat in df["treatment"].unique():
    fig = px.bar(
        table5[table5["treatment"] == treat],
        x="max_labor",
        y="labor",
        color="post",
        barmode="group",
        title=f"Figure 2: Labor Supply by Treatment ({treat})",
    )
    fig.show()  # TODO: might need to groupby to find mean by treatment, post, and max_labor

# %%
# Figure 4
# line plot with mean labor supply by period and treatment
fig = px.line(
    table5,
    x="post",
    y="labor",
    color="treatment",
    title="Figure 4: Mean Labor Supply by Period and Treatment",
)  #TODO: might need to find mean by period and treatment

# %%
# Regression results
# labor = constant  + post + treat + post*treat
# Table 6