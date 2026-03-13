#!/usr/bin/env python3
"""
Run updated Gruber & Saez observational study replication using PolicyEngine inputs.

Reads household-level broad income, taxable income, and marginal tax rates from
a PolicyEngine-generated CSV, queries the LLM for both taxable and broad income
responses under the new tax rate, and saves results to book/data/.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_eti.edsl_client_update import EDSLClient
from llm_eti.simulation_engine_update import SimulationParams, TaxSimulation

CSV_PATH = (
    Path(__file__).parent.parent.parent
    / "policy_engine_simulation"
    / "policyengine_sample_incomes.csv"
)


def main():
    parser = argparse.ArgumentParser(
        description="Run updated Gruber & Saez simulation from PolicyEngine inputs"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (1 household)"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument(
        "--responses", type=int, default=1, help="LLM responses per household"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"Error: input CSV not found at {CSV_PATH}")
        sys.exit(1)

    print(f"Running updated Gruber & Saez simulation with {args.model}...")
    print(f"Input CSV: {CSV_PATH}")

    client = EDSLClient(api_key=api_key, model=args.model, use_cache=True)
    params = SimulationParams(
        responses_per_household=args.responses,
        test_mode=args.test,
    )

    simulation = TaxSimulation(client, params)
    results_df = simulation.run_bulk_simulation(CSV_PATH)

    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    filename = f"gruber_saez_results_update_{args.model}"
    if args.test:
        filename += "_test"
    filename += ".csv"

    results_df.to_csv(output_dir / filename, index=False)
    print(f"Results saved to {output_dir / filename}")
    print(f"Total responses: {len(results_df)}")


if __name__ == "__main__":
    main()
