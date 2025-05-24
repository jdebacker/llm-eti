from simulation_engine import SimulationParams, TaxSimulation
from sklearn.linear_model import LinearRegression


def run_bulk_analysis(
    gpt_client, min_income=50000, max_income=200000, income_step=10000
):
    params = SimulationParams(
        min_rate=0.15, max_rate=0.35, step_size=0.02, responses_per_rate=10
    )

    simulation = TaxSimulation(gpt_client, params)
    results_df = simulation.run_bulk_simulation(
        min_income, max_income, income_step, 0.25
    )

    # Calculate average ETI by income level
    avg_etis = (
        results_df.groupby("Broad Income")["Implied ETI"].mean().reset_index()
    )

    # Run regression
    X = avg_etis[["Broad Income"]]
    y = avg_etis["Implied ETI"]
    reg = LinearRegression().fit(X, y)

    return {
        "results": results_df,
        "avg_etis": avg_etis,
        "regression": {
            "coefficient": reg.coef_[0],
            "intercept": reg.intercept_,
            "r2": reg.score(X, y),
        },
    }
