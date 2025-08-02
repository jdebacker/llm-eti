#!/usr/bin/env python3
"""Run minimal PKNF simulation with Gemini 2.5 Flash."""

import os
from llm_eti.edsl_client import EDSLClient
from llm_eti.simulation_engine import LabExperimentSimulation

def main():
    """Run minimal PKNF simulation."""
    
    if not os.getenv("EXPECTED_PARROT_API_KEY"):
        print("Error: EXPECTED_PARROT_API_KEY not set")
        return
    
    print("Running minimal PKNF simulation with gemini-2.5-flash...")
    
    # Initialize client
    client = EDSLClient(model="gemini-2.5-flash")
    sim = LabExperimentSimulation(client)
    
    # Run very minimal simulation
    print("\nRunning with:")
    print("- Treatment: Prog,Flat25 only")
    print("- Subjects: 2")
    print("- Rounds: 16 (8 pre-reform, 8 post-reform)")
    
    df = sim.run_experiment(
        treatments=["Prog,Flat25"],
        subjects_per_treatment=2,
        rounds=16
    )
    
    print(f"\nGot {len(df)} observations")
    
    if not df.empty:
        # Save results
        output_path = "book/data/pknf_results_gemini-2.5-flash_minimal.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        
        # Show results
        print("\nResults:")
        print(df[['subject_id', 'round', 'tax_schedule', 'labor_endowment', 'labor_supply']])
        
        # Basic stats
        print(f"\nAverage labor supply:")
        print(f"- Progressive tax: {df[df['tax_schedule'] == 'progressive']['labor_supply'].mean():.1f}")
        print(f"- Flat 25% tax: {df[df['tax_schedule'] == 'flat25']['labor_supply'].mean():.1f}")

if __name__ == "__main__":
    main()