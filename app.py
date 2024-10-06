import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gpt_utils import get_gpt4_response
from tax_utils import calculate_eti, parse_income_response
import os
from datetime import datetime

st.title("Tax Policy Simulation and ETI Calculator")

broad_income = st.number_input(
    "Prior Year's Broad Income", min_value=0.0, value=100000.0, step=1000.0
)
taxable_income = st.number_input(
    "Prior Year's Taxable Income (after deductions)",
    min_value=0.0,
    value=80000.0,
    step=1000.0,
)
prior_marginal_rate = st.number_input(
    "Prior Year's Marginal Tax Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.01,
)

min_rate = st.number_input(
    "Minimum Simulated Tax Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.01,
)
max_rate = st.number_input(
    "Maximum Simulated Tax Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01,
)
step_size = st.number_input(
    "Step Size", min_value=0.01, max_value=0.1, value=0.02, step=0.01
)
n = st.number_input(
    "Number of Responses per Tax Rate", min_value=1, value=10, step=1
)

if st.button("Run Simulation"):
    tax_rates = np.arange(min_rate, max_rate + step_size, step_size)
    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for rate in tax_rates:
        prompt = f"""
        You are a taxpayer with the following tax profile:
        - Your broad income last year was ${broad_income:,.2f}
        - Your taxable income last year (after deductions) was ${taxable_income:,.2f}
        - Your marginal tax rate last year was {prior_marginal_rate:.2%}
        
        This year, if you had the same broad income as last year, your marginal tax rate will change to {rate:.2%}. Given this change, estimate your taxable income for this year.
        Consider how this change in tax rate might affect your behavior, such as your work hours, investment decisions, or tax planning strategies.
        
        Provide your response as a single number representing your estimated taxable income for this year, rounded to the nearest dollar. Do not include any explanations or additional text.
        """

        responses = get_gpt4_response(prompt, n=n)

        new_taxable_income_sum = 0
        valid_responses = 0
        for response in responses:
            parsed_income = parse_income_response(response)
            if parsed_income is not None:
                new_taxable_income_sum += parsed_income
                valid_responses += 1

        if valid_responses > 0:
            average_new_taxable_income = (
                new_taxable_income_sum / valid_responses
            )
            eti = calculate_eti(
                prior_marginal_rate,
                rate,
                taxable_income,
                average_new_taxable_income,
            )
            results.append(
                {
                    "Timestamp": timestamp,
                    "Broad Income": broad_income,
                    "Prior Taxable Income": taxable_income,
                    "Prior Marginal Rate": prior_marginal_rate,
                    "New Tax Rate": rate,
                    "Average Expected Taxable Income": average_new_taxable_income,
                    "Implied ETI": eti,
                    "Valid Responses": valid_responses,
                }
            )
        else:
            st.warning(
                f"No valid responses for tax rate {rate:.2%}. Skipping this data point."
            )

    if results:
        df = pd.DataFrame(results)
        st.write(df)

        # Append results to CSV
        csv_filename = "tax_simulation_results.csv"
        df.to_csv(
            csv_filename,
            mode="a",
            header=not os.path.exists(csv_filename),
            index=False,
        )
        st.success(f"Results appended to {csv_filename}")

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(
            df["New Tax Rate"],
            df["Average Expected Taxable Income"],
            color="blue",
            label="Average Expected Taxable Income",
        )
        ax2.plot(
            df["New Tax Rate"],
            df["Implied ETI"],
            color="red",
            label="Implied ETI",
        )

        ax1.set_xlabel("Marginal Tax Rate")
        ax1.set_ylabel("Expected Taxable Income ($)", color="blue")
        ax2.set_ylabel("Implied ETI", color="red")

        ax1.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title(
            "Expected Taxable Income and Implied ETI vs. Marginal Tax Rate"
        )
        fig.legend(
            loc="upper right",
            bbox_to_anchor=(1, 1),
            bbox_transform=ax1.transAxes,
        )

        st.pyplot(fig)

        avg_eti = df["Implied ETI"].mean()
        st.write(f"Average Implied ETI: {avg_eti:.4f}")
    else:
        st.error("No valid results were obtained from the simulation.")

st.write(
    "Note: This simulation uses GPT-4 to estimate expected taxable income. Results may vary and should be interpreted with caution."
)
