import click
import json
from datetime import datetime
from pathlib import Path
from llm_eti.config import Config
from llm_eti.simulation_engine import SimulationParams, TaxSimulation
from llm_eti.gpt_utils import GPTClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return super().default(obj)


@click.group()
def cli():
    """Tax Policy Simulation CLI"""
    pass


def calculate_summary_statistics(df):
    """Calculate summary statistics from simulation results"""
    # Calculate stats by income level
    stats_by_income = {}
    for income in sorted(df["broad_income"].unique()):
        income_df = df[df["broad_income"] == income]
        stats_by_income[str(int(income))] = {
            "eti_mean": float(income_df["implied_eti"].mean()),
            "eti_std": float(income_df["implied_eti"].std()),
            "response_count": int(income_df["implied_eti"].count()),
            "income_mean": float(income_df["parsed_income"].mean()),
            "income_std": float(income_df["parsed_income"].std()),
        }

    return {
        "overall_avg_eti": float(df["implied_eti"].mean()),
        "overall_std_eti": float(df["implied_eti"].std()),
        "by_income": stats_by_income,
        "sample_size": int(len(df)),
        "valid_responses": int(df["parsed_income"].notna().sum()),
    }


def generate_visualizations(df, output_dir):
    """Generate and save visualizations"""
    plt.style.use("default")

    # ETI by Income Level
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="broad_income", y="implied_eti")
    plt.xticks(rotation=45)
    plt.title("ETI Distribution by Income Level")
    plt.tight_layout()
    plt.savefig(output_dir / "eti_by_income.png")
    plt.close()

    # ETI by Tax Rate
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="new_rate", y="implied_eti", hue="broad_income")
    plt.title("ETI vs Tax Rate by Income Level")
    plt.tight_layout()
    plt.savefig(output_dir / "eti_by_tax_rate.png")
    plt.close()


@cli.command()
@click.option(
    "--min-income",
    default=Config.DEFAULT_PARAMS["min_income"],
    help="Minimum income to simulate",
)
@click.option(
    "--max-income",
    default=Config.DEFAULT_PARAMS["max_income"],
    help="Maximum income to simulate",
)
@click.option(
    "--income-step",
    default=Config.DEFAULT_PARAMS["income_step"],
    help="Income step size",
)
@click.option(
    "--model", default=Config.DEFAULT_PARAMS["model"], help="GPT model to use"
)
@click.option(
    "--output",
    default=None,
    help="Output directory name (defaults to timestamp)",
)
@click.option(
    "--rate-step",
    default=Config.DEFAULT_PARAMS["rate_step"],
    help="Tax rate step size",
)
@click.option(
    "--responses-per-rate",
    default=Config.DEFAULT_PARAMS["responses_per_rate"],
    help="Number of responses per tax rate",
)
def run_simulation(
    min_income,
    max_income,
    income_step,
    model,
    output,
    rate_step,
    responses_per_rate,
):
    """Run bulk tax policy simulation"""

    if not Config.OPENAI_API_KEY:
        raise click.ClickException(
            "OPENAI_API_KEY not found in environment variables"
        )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.RESULTS_DIR / (output or f"simulation_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Initialize client and parameters
    client = GPTClient(api_key=Config.OPENAI_API_KEY, model=model)
    params = SimulationParams(
        min_rate=Config.DEFAULT_PARAMS["min_rate"],
        max_rate=Config.DEFAULT_PARAMS["max_rate"],
        step_size=rate_step,
        responses_per_rate=responses_per_rate,
        taxable_income_ratio=Config.DEFAULT_PARAMS["taxable_income_ratio"],
    )

    # Run simulation
    click.echo(f"\nRunning simulation with model {model}...")
    click.echo(
        f"Income range: ${min_income:,} to ${max_income:,} in steps of ${income_step:,}"
    )
    click.echo(
        f"Tax rate range: {params.min_rate:.1%} to {params.max_rate:.1%} in steps of {params.step_size:.1%}"
    )
    click.echo(f"Responses per rate: {params.responses_per_rate}\n")

    simulation = TaxSimulation(client, params)
    results_df = simulation.run_bulk_simulation(
        min_income,
        max_income,
        income_step,
        Config.DEFAULT_PARAMS["prior_rate"],
    )

    if results_df.empty:
        click.echo("No results generated. Please check for errors above.")
        return

    # Save raw results
    results_path = output_dir / "raw_responses.csv"
    results_df.to_csv(results_path, index=False)
    click.echo(f"\nRaw responses saved to {results_path}")

    # Calculate and save summary statistics
    summary = calculate_summary_statistics(results_df)
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    click.echo(f"Summary statistics saved to {summary_path}")

    # Generate visualizations
    generate_visualizations(results_df, output_dir)
    click.echo(f"Visualizations saved to {output_dir}")

    # Run regression analysis
    from analysis import analyze_eti_heterogeneity

    reg_results = analyze_eti_heterogeneity(results_df, output_dir)
    click.echo(f"Regression table saved to {output_dir}/regression_table.tex")

    # Print summary
    click.echo("\nSummary Statistics:")
    click.echo(f"Average ETI: {summary['overall_avg_eti']:.3f}")
    click.echo(f"Standard Deviation: {summary['overall_std_eti']:.3f}")
    click.echo(f"Total responses attempted: {summary['sample_size']}")
    click.echo(f"Valid responses: {summary['valid_responses']}")


if __name__ == "__main__":
    cli()
