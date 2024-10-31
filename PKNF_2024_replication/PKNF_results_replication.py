# imports
import pandas as pd
import numpy as np
import linearmodels as lm
import os
import pickle
import plotly.express as px


# Read in data
DATA_FILENAME = "PKNF_replication_results"
df = pickle.load(open(os.path.join("..", "data", DATA_FILENAME + ".pkl"), "wb"))

# Parse model results to get chose income
df["income"] = df["model_answer"].str.extract(r"Chose income: (\d+)")
# Create a pre-post variable
df['post'] = df['round_num'] >= 8
# create labor supply variable
df["labor"] = df["income"].astype(int) / 20
df["labor_20"] = df["labor"] <= 20

# Create Table 5
table5 = df.groupby(["treatment", "post"]).mean("labor_20").reset_index()
# TODO: add p-value for test of diffs in means

# Figure 4

