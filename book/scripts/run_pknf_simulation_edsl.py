#!/usr/bin/env python3
"""
Run PKNF (2024) lab experiment replication using EDSL.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_eti.edsl_client import EDSLClient
from llm_eti.simulation_engine_edsl import LabExperimentSimulation


def main():
    parser = argparse.ArgumentParser(description="Run PKNF simulation with EDSL")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with fewer subjects"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("EXPECTED_PARROT_API_KEY")
    if not api_key:
        print("Error: EXPECTED_PARROT_API_KEY environment variable not set")
        sys.exit(1)

    # Set parameters
    num_subjects = 10 if args.test else 100
    
    # All treatments from PKNF
    treatments = [
        "Prog,Prog",
        "Prog,Flat25", 
        "Prog,Flat50",
        "Flat25,Prog",
        "Flat50,Prog"
    ]

    print(f"Running PKNF simulation with {args.model} ({num_subjects} subjects per treatment)...")
    print("Using EDSL with universal cache enabled")

    # Initialize client and experiment
    client = EDSLClient(api_key=api_key, model=args.model, use_cache=True)
    experiment = LabExperimentSimulation(client)
    
    # Run experiment
    results_df = experiment.run_experiment(
        treatments=treatments,
        rounds=16,
        subjects_per_treatment=num_subjects
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    filename = f"pknf_results_{args.model}_edsl"
    if args.test:
        filename += "_test"

    results_df.to_csv(output_dir / f"{filename}.csv", index=False)
    print(f"Results saved to {output_dir / f'{filename}.csv'}")
    print(f"Total responses: {len(results_df)}")


if __name__ == "__main__":
    main()