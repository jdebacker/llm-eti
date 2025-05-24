# %%
# imports
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import pickle
import plotly.express as px
from matplotlib import pyplot as plt


# Define parameters for matplotlib plots
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#d0d0d0'
plt.rcParams['grid.color'] = '#d0d0d0'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.axis'] = 'y'  # Only show horizontal grid lines

# get directory of current file
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# set plotly to use white theme
px.defaults.template = "plotly_white"

# %%
# Read in data
DATA_FILENAME = "PKNF_replication_results_gpt-4o-mini"
df_raw = pickle.load(
    open(os.path.join(CUR_DIR, "..", "data", DATA_FILENAME + ".pkl"), "rb")
)

# %%
def clean_data(df):
    """
    Applies some cleaning to the data
    """
    print("Cleaning data...")
    # Parse model results to get chose income
    wage_rate = 20
    df["income"] = df["chosen_labor"] * wage_rate
    # drop if missing income
    df = df.dropna(subset=["income"])
    # drop if chosen_labor is < 1
    df = df[df["chosen_labor"] >= 1]
    # Make labor was written in terms of max income
    df["max_labor"] = df["max_labor"] / wage_rate

    # Create a pre-post variable
    df["Post"] = (df["round"] >= 8).astype(int)
    # create labor supply variable
    df["labor"] =  df["chosen_labor"]
    df["labor_20"] = (df["labor"] <= 20).astype(int)
    df["lab_supply"] = df["labor"] / df["max_labor"]
    # drop if labor supply > 1, this is not possible and likely error parsing data
    df = df[df["lab_supply"] <= 1]

    return df

df = clean_data(df_raw)

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
table5.to_markdown(
    os.path.join(CUR_DIR, "tables_figures", "table5.md"), floatfmt=".2f", index=False
)
table5.to_latex(
    os.path.join(CUR_DIR, "tables_figures", "table5.tex"),
    float_format="%.2f",
    index=False,
)
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
        nbins=len(list(df_bar["max_labor"].unique())) + 2,
        # title=f"Figure 2: Labor Supply by Treatment ({treat})",
    )
    # Specify the x- and y-axis labels
    fig.update_xaxes(title_text="Maximum Labor")
    fig.update_yaxes(title_text="Labor Supply, in Units")
    # set x ticks
    fig.update_xaxes(tickvals=list(df_bar["max_labor"].unique()))
    fig.write_image(
        os.path.join(CUR_DIR, "tables_figures", f"LLM_Fig2_{treat_num}.png")
    )
#############################
# Make same figure as above, but in Matplotlib
#############################
max_labor_values = sorted(df_bar["max_labor"].unique())
post_values = sorted(df_bar["Post"].unique())
colors = ["#f8953a", "#4c72b0"]
bar_width = 0.35
gap = 0.05  # gap between groups
for treat in df_bar["Treatment"].unique():
    # Set up the figure
    fig, ax = plt.subplots()
    for i, max_lab in enumerate(max_labor_values):
        group_center = i
        for j, post in enumerate(post_values):
            pos = group_center - (bar_width + gap)/2 + j*(bar_width + gap)
            df_treat = df_bar[(df_bar["max_labor"] == max_lab) &
                            (df_bar["Post"] == post) &
                            (df_bar["Treatment"] == treat)]
            if not df_treat.empty:
                ax.bar(pos,
                    df_treat["labor"].values[0],
                    width=bar_width,
                    color=colors[j],
                    alpha=0.9 if post == 0 else 1.0,
                    label=f"Post" if post == 1 and i==0 else
                            f"Pre" if i==0 else "",
                    edgecolor='black',
                    linewidth=0.5)
    # Add dashed line for the mean
    mean_value = df_bar[df_bar["Treatment"] == treat]["labor"].mean()
    ax.axhline(y=mean_value, color='#808080', linestyle='--', linewidth=1.5, alpha=0.8)
    # Add max value line
    max_value = df_bar[df_bar["Treatment"] == treat]["labor"].max()
    ax.axhline(y=mean_value, color='#808080', linestyle=':', linewidth=1.5, alpha=0.3)
    ax.set_xlabel("Maximum Labor")
    ax.set_ylabel("Labor Supply, in Units")
    #title
    ax.set_title(f"{treat}")
    # set y_lim
    ax.set_ylim(0, max_value + 0.5)
    ax.set_xticks(np.arange(len(max_labor_values)))
    ax.set_xticklabels(max_labor_values)
    ax.legend()

    plt.savefig(
        os.path.join(CUR_DIR, "tables_figures", f"LLM_Fig2_{treat}_matplotlib.png"),
    bbox_inches="tight",
    dpi=300,
)

# %%
# Figure 4
# line plot with mean labor supply by period and treatment
# First, groupby to find mean by treatment, period
df_bar = (
    df[["treatment", "lab_supply", "round"]]
    .groupby(["treatment", "round"])
    .mean()
    .reset_index()
)
# update treatment to be more descriptive
df_bar["Treatment"] = df_bar["treatment"].map(treat_dict)
fig = px.scatter(
    df_bar,
    x="round",
    y="lab_supply",
    color="Treatment",
    # title="Figure 4: Mean Labor Supply by Period and Treatment",
).update_traces(mode="lines+markers")
# Specify the x- and y-axis labels
# set y axis limits
fig.update_layout(yaxis_range=[0.7, 1.02])
fig.update_xaxes(title_text="Period")
fig.update_yaxes(title_text="Labor Supply in %")
# put a vertical dashed line at period 8
fig.add_vline(x=7.5, line_dash="dash")
# fig.show()
fig.write_image(os.path.join(CUR_DIR, "tables_figures", "LLM_Fig4.png"))

###########################
# Same figure as above, but in Matplotlib
###########################
# Set up the figure
fig, ax = plt.subplots()
for treat in df_bar["Treatment"].unique():
    # Filter the data for the current treatment
    df_treat = df_bar[df_bar["Treatment"] == treat]
    # Plot the data
    ax.plot(
        df_treat["round"],
        df_treat["lab_supply"],
        marker="o",
        label=treat,
        alpha=0.9,
    )
# set y axis limits
ax.set_ylim(0.7, 1.02)
# Add a vertical dashed line at period 8
ax.axvline(x=8, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
# Add labels and title
ax.set_xlabel("Period")
ax.set_ylabel("Labor Supply in %")
# ax.set_title("Figure 4: Mean Labor Supply by Period and Treatment")
# Add legend
ax.legend()
# Save the figure
plt.savefig(
    os.path.join(CUR_DIR, "tables_figures", "LLM_Fig4_matplotlib.png"),
    bbox_inches="tight",
    dpi=300,
)

# %%
# Regression results

PKNF_Table6_Col1 = [
    "-0.007",
    "(0.006)",
    "-0.052",
    "(0.022)",
    "0.083",
    "(0.009)",
    "0.947",
    "(0.055)",
    "3344",
    "0.035",
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
    model = smf.ols("lab_supply ~ Post + treated + post_treat", data=df_reg)
    # Estimate the model
    results = model.fit()
    reg_results_dict[treat] = []  # initialize list to store results
    reg_results_dict[treat].append(f"{results.params["Post"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["Post"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["treated"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["treated"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["post_treat"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["post_treat"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["Intercept"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["Intercept"]:.3f})")
    reg_results_dict[treat].extend([f"{results.nobs:.0f}", f"{results.rsquared:.3f}"])
# put into dataframe
reg_results = pd.DataFrame(reg_results_dict)
reg_results.index = [
    "Post",
    "",
    "Treated",
    "",
    "Post*Treated",
    "",
    "Constant",
    "",
    "N",
    "R-squared",
]
# Save as markdown
reg_results.to_markdown(
    os.path.join(CUR_DIR, "tables_figures", "table6.md"), index=True
)
reg_results.to_latex(os.path.join(CUR_DIR, "tables_figures", "table6.tex"), index=True)

# Make table comparing to PKNF results to LLM results
compare_reg_dict = {"PKNF": PKNF_Table6_Col1, "LLM": reg_results["Prog,Flat25"]}
reg_results = pd.DataFrame(compare_reg_dict)
reg_results.index = [
    "Post",
    "",
    "Treated",
    "",
    "Post*Treated",
    "",
    "Constant",
    "",
    "N",
    "R-squared",
]
# Save as markdown
reg_results.to_markdown(
    os.path.join(CUR_DIR, "tables_figures", "table6_compare.md"), index=True
)
reg_results.to_latex(
    os.path.join(CUR_DIR, "tables_figures", "table6_compare.tex"), index=True
)


# %% Use
# Use regression model to estimate the ETI
# do regression as above, but with log income as the dependent variable
# then tax coefficient on the DD and divide by the change in (1-tau)
# e.g. for the Prog,Flat25 group, this is (0.5 - 0.25)
df["log_income"] = np.log(df["income"])
# drop if missing or infinite log income
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
reg_results_dict = {}
for treat in ["Prog,Flat25", "Prog,Flat50", "Flat25,Prog", "Flat50,Prog"]:
    # Keep the treatment and control
    df_reg = df[(df["Treatment"] == treat) | (df["treatment"] == 1)]
    # specify an indicator for the treated group
    model = smf.ols("log_income ~ Post + treated + post_treat", data=df_reg)
    # Estimate the model
    results = model.fit()
    reg_results_dict[treat] = []  # initialize list to store results
    reg_results_dict[treat].append(f"{results.params["Post"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["Post"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["treated"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["treated"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["post_treat"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["post_treat"]:.3f})")
    reg_results_dict[treat].append(f"{results.params["Intercept"]:.3f}")
    reg_results_dict[treat].append(f"({results.bse["Intercept"]:.3f})")
    reg_results_dict[treat].extend([f"{results.nobs:.0f}", f"{results.rsquared:.3f}"])

# put into dataframe
reg_results = pd.DataFrame(reg_results_dict)
reg_results.index = [
    "Post",
    "",
    "Treated",
    "",
    "Post*Treated",
    "",
    "Constant",
    "",
    "N",
    "R-squared",
]
# Save as markdown
reg_results.to_markdown(
    os.path.join(CUR_DIR, "tables_figures", "income_reg_table.md"), index=True
)
reg_results.to_latex(
    os.path.join(CUR_DIR, "tables_figures", "income_reg_table.tex"), index=True
)

# %%
# Compute ETI
ETI = reg_results.loc["Post*Treated", :].astype(float) / np.array(
    [0.25, 0.1, -0.25, -0.1]
)

# %%
# ETI regressions
#
mtr25_df1 = df[(df["treatment"] == 2) & (df["Post"] == 1)]
mtr25_df1["tau"] = 0.75
mtr25_df2 = df[(df["treatment"] == 4) | (df["Post"] == 0)]
mtr25_df2["tau"] = 0.75
mtr50_df1 = df[(df["treatment"] == 3) & (df["Post"] == 1)]
mtr50_df1["tau"] = 0.5
mtr50_df2 = df[(df["treatment"] == 5) | (df["Post"] == 0)]
mtr50_df2["tau"] = 0.5
mtr_df = pd.concat([mtr25_df1, mtr25_df2, mtr50_df1, mtr50_df2])
model = smf.ols("log_income ~ tau", data=mtr_df)
results = model.fit()
print(results.summary())


# %%
# bunching plots
# plot histogram by earnings for treatment 2
fig = px.histogram(
    df[df["treatment"] == 2],
    x="income",
    color="Post",
    nbins=100,
    # title="Histogram of Earnings for Treatment 2",
)
# update x-axis label
fig.update_xaxes(title_text="Pre-tax Income")
fig.update_yaxes(title_text="Count")
# fig.show()
fig.write_image(os.path.join(CUR_DIR, "tables_figures", "LLM_bunching.png"))

##############
# Same figure as above, but in Matplotlib
##############
# Set up the figure
fig, ax = plt.subplots()
# Filter the data for treatment 2
df_treat = df[df["treatment"] == 2]
# Plot the data
ax.hist(
    df_treat[df_treat["Post"] == 0]["income"],
    bins=20,
    alpha=0.5,
    density=True,
    label="Prog",
    color="#f8953a",
)
ax.hist(
    df_treat[df_treat["Post"] == 1]["income"],
    bins=20,
    alpha=0.5,
    density=True,
    label="Flat25",
    color="#4c72b0",
)
# Add labels and title
ax.set_xlabel("Pre-tax Income")
ax.set_ylabel("Density")
# ax.set_title("Histogram of Earnings for Treatment 1")
# Add legend
ax.legend()
# Save the figure
plt.savefig(
    os.path.join(CUR_DIR, "tables_figures", "LLM_bunching_matplotlib.png"),
    bbox_inches="tight",
    dpi=300,
)

# %%
# create datasets for bunching estimation in R
# There will be one CSV file per model and per 40/50 top rate
# rows will be chosen_labor values, columns will be the treatment
for model in ["claude-3-haiku-20240307", "gpt-4.1-mini-2025-04-14", "gpt-4o-mini"]:
    filename= f"PKNF_modified_40pct_results_{model}.csv"
    df = pd.read_csv(
        os.path.join(CUR_DIR, "..", "data", filename)
    )
    df = clean_data(df.copy())
    df = df[["treatment", "Post", "income"]]
    # df_counts = df.groupby(["treatment", "Post", "income"])["subject_id"].count().reset_index()
    # # rename subject_id to counts
    # df_counts = df_counts.rename(columns={"subject_id": "counts"})

    df.to_csv(
        os.path.join(CUR_DIR, "..", "data", f"PKNF_40_bunching_{model}.csv"),
        index=False,
    )
    # make a bunching plot for each treatment
    # Set up the figure
    fig, ax = plt.subplots()
    # Filter the data for treatment 2
    df_treat = df[df["treatment"] == 2]
    # Plot the data
    ax.hist(
        df_treat[df_treat["Post"] == 0]["income"],
        bins=20,
        alpha=0.5,
        density=True,
        label="Prog",
        color="#f8953a",
    )
    ax.hist(
        df_treat[df_treat["Post"] == 1]["income"],
        bins=20,
        alpha=0.5,
        density=True,
        label="Flat25",
        color="#4c72b0",
    )
    # Add labels and title
    ax.set_xlabel("Pre-tax Income")
    ax.set_ylabel("Density")
    # ax.set_title("Histogram of Earnings for Treatment 1")
    # Add legend
    ax.legend()
    # Save the figure
    plt.savefig(
        os.path.join(CUR_DIR, "tables_figures", f"LLM_bunching_40pct_{model}_matplotlib.png"),
        bbox_inches="tight",
        dpi=300,
    )

# do for PKNF with 50% top rate...

# %%
