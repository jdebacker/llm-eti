#!/usr/bin/env python3
"""Run PKNF simulation with Gemini 2.5 Flash."""

import os
from llm_eti.edsl_client import EDSLClient
from llm_eti.simulation_engine import LabExperimentSimulation
from llm_eti.pknf_analysis import run_did_analysis, calculate_bunching_eti

def main():
    """Run PKNF simulation with Gemini 2.5 Flash."""
    
    # Check API key
    if not os.getenv("EXPECTED_PARROT_API_KEY"):
        print("Error: EXPECTED_PARROT_API_KEY not set")
        return
    
    print("Running PKNF simulation with gemini-2.5-flash...")
    print("=" * 60)
    
    # Initialize client
    client = EDSLClient(model="gemini-2.5-flash")
    sim = LabExperimentSimulation(client)
    
    # Run test simulation (smaller scale)
    print("\nRunning test simulation:")
    print("- Treatments: 2 (Prog,Flat25 and Flat25,Prog)")
    print("- Subjects per treatment: 5")
    print("- Rounds: 16")
    
    df = sim.run_experiment(
        treatments=["Prog,Flat25", "Flat25,Prog"],
        subjects_per_treatment=5,
        rounds=16
    )
    
    print(f"\nCompleted! Got {len(df)} total observations")
    
    # Save results
    output_path = "book/data/pknf_results_gemini-2.5-flash_test.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Basic analysis
    print("\nBasic Statistics:")
    print(f"- Average labor supply: {df['labor_supply'].mean():.2f}")
    print(f"- Labor supply by tax schedule:")
    for schedule in df['tax_schedule'].unique():
        avg = df[df['tax_schedule'] == schedule]['labor_supply'].mean()
        print(f"  - {schedule}: {avg:.2f}")
    
    # DiD analysis
    print("\nDifference-in-Differences Analysis:")
    did_results = run_did_analysis(df)
    print(did_results.to_string(index=False))
    
    # ETI calculation
    print(f"\nETI Lower Bound: {calculate_bunching_eti(df):.3f}")

if __name__ == "__main__":
    main()