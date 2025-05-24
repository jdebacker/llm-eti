import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant


def run_model_regressions(model_df: pd.DataFrame, model_name: str) -> dict:
    """Run three regressions for a single model."""
    print(f"\nRunning regression for {model_name}")
    print(f"N = {len(model_df)}")

    try:
        # Regression 1: ETI ~ Income
        X1 = add_constant(model_df[["income_100k"]])
        reg1 = sm.OLS(model_df["implied_eti"], X1).fit(cov_type="HC1")
        print(reg1.summary())

        # Regression 2: ETI ~ Income + Absolute MTR change
        X2 = add_constant(model_df[["income_100k", "abs_mtr_change"]])
        reg2 = sm.OLS(model_df["implied_eti"], X2).fit(cov_type="HC1")
        print(reg2.summary())

        # Regression 3: Add interaction
        model_df["income_abs_mtr_interact"] = (
            model_df["income_100k"] * model_df["abs_mtr_change"]
        )
        X3 = add_constant(
            model_df[
                ["income_100k", "abs_mtr_change", "income_abs_mtr_interact"]
            ]
        )
        reg3 = sm.OLS(model_df["implied_eti"], X3).fit(cov_type="HC1")
        print(reg3.summary())

        print(
            f"R-squared values: {reg1.rsquared:.3f}, {reg2.rsquared:.3f}, {reg3.rsquared:.3f}"
        )

        return {"Model": model_name, "regs": [reg1, reg2, reg3]}

    except Exception as e:
        print(f"Error in regression for {model_name}: {str(e)}")
        return None
