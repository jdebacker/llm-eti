#!/usr/bin/env python3
"""
Run Gruber & Saez (2002) observational study replication using LLMs.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent.parent))

from simulation_engine import TaxSimulation
from gpt_utils import GPTClient
from config import SimulationParams


def main():
    parser = argparse.ArgumentParser(description='Run Gruber & Saez simulation')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with fewer scenarios')
    parser.add_argument('--model', default='gpt-4o-mini',
                        help='LLM model to use')
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print(f"Running Gruber & Saez simulation with {args.model}...")
    
    # Initialize client and parameters
    client = GPTClient(api_key=api_key, model=args.model)
    params = SimulationParams(
        responses_per_rate=10 if args.test else 100,
        min_rate=0.15,
        max_rate=0.35,
        step_size=0.02
    )
    
    # Set income range
    min_income = 50000
    max_income = 80000 if args.test else 200000
    income_step = 30000 if args.test else 10000
    
    # Run simulation
    simulation = TaxSimulation(client, params)
    results_df = simulation.run_bulk_simulation(
        min_income=min_income,
        max_income=max_income,
        income_step=income_step,
        prior_rate=0.25
    )
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    filename = f'gruber_saez_results_{args.model}'
    if args.test:
        filename += '_test'
    
    results_df.to_csv(output_dir / f'{filename}.csv', index=False)
    print(f"Results saved to {output_dir / f'{filename}.csv'}")


if __name__ == '__main__':
    main()