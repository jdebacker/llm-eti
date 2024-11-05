import streamlit as st
import matplotlib.pyplot as plt
from simulation_engine import SimulationParams, TaxSimulation


def run_streamlit_app(gpt_client):
    st.title("Tax Policy Simulation and ETI Calculator")

    # Input parameters
    params = SimulationParams(
        min_rate=st.number_input(
            "Minimum Tax Rate", min_value=0.0, max_value=1.0, value=0.15
        ),
        max_rate=st.number_input(
            "Maximum Tax Rate", min_value=0.0, max_value=1.0, value=0.35
        ),
        step_size=st.number_input(
            "Step Size", min_value=0.01, max_value=0.1, value=0.02
        ),
        responses_per_rate=st.number_input(
            "Responses per Rate", min_value=1, value=10
        ),
    )

    broad_income = st.number_input(
        "Prior Year's Broad Income", min_value=0.0, value=100000.0
    )
    prior_rate = st.number_input(
        "Prior Year's Marginal Rate", min_value=0.0, max_value=1.0, value=0.25
    )

    if st.button("Run Simulation"):
        simulation = TaxSimulation(gpt_client, params)
        df = simulation.run_bulk_simulation(
            broad_income, broad_income, 1, prior_rate
        )

        # Display results and create visualizations
        st.write(df)
        create_visualization(df)


def create_visualization(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Expected Taxable Income vs Tax Rate
    df.plot(x="New Tax Rate", y="Average Expected Taxable Income", ax=ax1)
    ax1.set_title("Expected Taxable Income vs Tax Rate")
    ax1.set_xlabel("Marginal Tax Rate")
    ax1.set_ylabel("Expected Taxable Income ($)")

    # Plot 2: Implied ETI vs Tax Rate
    df.plot(x="New Tax Rate", y="Implied ETI", ax=ax2)
    ax2.set_title("Implied ETI vs Tax Rate")
    ax2.set_xlabel("Marginal Tax Rate")
    ax2.set_ylabel("Implied ETI")

    plt.tight_layout()
    st.pyplot(fig)
