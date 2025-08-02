#!/usr/bin/env python3
"""
Run PKNF (2024) lab experiment replication using LLMs.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent.parent))

from PKNF_2024_replication.GPT_PKNF_replication import TaxBehaviorReplication


def main():
    parser = argparse.ArgumentParser(description='Run PKNF simulation')
    parser.add_argument('--test', action='store_true', 
                        help='Run in test mode with fewer subjects')
    parser.add_argument('--model', default='gpt-4o-mini',
                        help='LLM model to use')
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Set parameters
    num_subjects = 10 if args.test else 100
    
    print(f"Running PKNF simulation with {args.model} ({num_subjects} subjects)...")
    
    # Initialize and run experiment
    experiment = TaxBehaviorReplication(model=args.model)
    results_df = experiment.run_full_experiment(num_subjects=num_subjects)
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    filename = f'pknf_results_{args.model}'
    if args.test:
        filename += '_test'
    
    results_df.to_csv(output_dir / f'{filename}.csv', index=False)
    print(f"Results saved to {output_dir / f'{filename}.csv'}")


if __name__ == '__main__':
    main()