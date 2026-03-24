#!/usr/bin/env python3
"""
Run Gruber & Saez (2002) observational study replication using EDSL.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_eti.edsl_client import EDSLClient
from llm_eti.simulation_engine import SimulationParams, TaxSimulation


def main():
    parser = argparse.ArgumentParser(
        description="Run Gruber & Saez simulation with EDSL"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with fewer scenarios"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument(
        "--production", action="store_true", help="Run full production simulation"
    )
    parser.add_argument(
        "--cache-analysis", action="store_true", help="Analyze cache usage after run"
    )
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("EXPECTED_PARROT_API_KEY")
    if not api_key:
        print("Error: EXPECTED_PARROT_API_KEY environment variable not set")
        sys.exit(1)

    print(f"Running Gruber & Saez simulation with {args.model} using EDSL...")

    # Initialize client and parameters
    client = EDSLClient(api_key=api_key, model=args.model, use_cache=True)

    # Set parameters based on mode
    if args.test:
        responses_per_rate = 10
        min_income, max_income, income_step = 50000, 80000, 30000
    elif args.production:
        responses_per_rate = 100
        min_income, max_income, income_step = 50000, 200000, 10000
    else:
        # Default mode - medium size
        responses_per_rate = 50
        min_income, max_income, income_step = 50000, 150000, 20000

    params = SimulationParams(
        responses_per_rate=responses_per_rate,
        min_rate=0.15,
        max_rate=0.35,
        step_size=0.02,
    )

    print("Running simulation with:")
    print(f"  - Model: {args.model}")
    print(f"  - Responses per rate: {responses_per_rate}")
    print(
        f"  - Income range: ${min_income:,} to ${max_income:,} (step ${income_step:,})"
    )
    print(f"  - Cache enabled: {client.use_cache}")

    # Run simulation
    simulation = TaxSimulation(client, params)
    results_df = simulation.run_bulk_simulation(
        min_income=min_income,
        max_income=max_income,
        income_step=income_step,
        prior_rate=0.25,
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    filename = f"gruber_saez_results_{args.model}"
    if args.test:
        filename += "_test"

    results_df.to_csv(output_dir / f"{filename}.csv", index=False)
    print(f"Results saved to {output_dir / f'{filename}.csv'}")
    print(f"Total responses: {len(results_df)}")
    print(f"Cache usage enabled: {client.use_cache}")

    # Analyze cache if requested
    if args.cache_analysis:
        from llm_eti.cache_utils import CacheExplorer

        print("\nCache Analysis:")
        explorer = CacheExplorer()
        stats = explorer.get_cache_stats()

        if "error" not in stats:
            print(f"  - Total cache entries: {stats['total_entries']}")
            print(f"  - Models in cache: {list(stats['models'].keys())}")

            savings = explorer.estimate_cost_savings()
            if "error" not in savings:
                print(
                    f"  - Estimated cost saved: ${savings['estimated_cost_saved']:.4f}"
                )


if __name__ == "__main__":
    main()
