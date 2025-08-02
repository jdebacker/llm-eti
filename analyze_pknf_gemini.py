#!/usr/bin/env python3
"""Analyze PKNF results from Gemini."""

import pandas as pd
from llm_eti.pknf_analysis import run_did_analysis, calculate_bunching_eti

def main():
    """Analyze PKNF results."""
    
    # Load results
    df = pd.read_csv("book/data/pknf_results_gemini-2.5-flash_minimal.csv")
    
    print("PKNF Experiment Results - Gemini 2.5 Flash")
    print("=" * 50)
    
    # Basic stats
    print(f"\nTotal observations: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Rounds: {df['round'].nunique()}")
    
    # Average labor supply by tax schedule
    print("\nAverage labor supply by tax schedule:")
    for schedule in df['tax_schedule'].unique():
        avg = df[df['tax_schedule'] == schedule]['labor_supply'].mean()
        print(f"- {schedule}: {avg:.2f}")
    
    # Check for bunching at 20 hours (income of 400)
    print("\nLabor supply = 20 (income = 400) analysis:")
    for schedule in df['tax_schedule'].unique():
        schedule_df = df[df['tax_schedule'] == schedule]
        at_20 = (schedule_df['labor_supply'] == 20).sum()
        total = len(schedule_df)
        pct = (at_20 / total) * 100
        print(f"- {schedule}: {at_20}/{total} = {pct:.1f}%")
    
    # Labor utilization
    df['labor_utilization'] = df['labor_supply'] / df['labor_endowment']
    print("\nAverage labor utilization:")
    for schedule in df['tax_schedule'].unique():
        avg_util = df[df['tax_schedule'] == schedule]['labor_utilization'].mean()
        print(f"- {schedule}: {avg_util:.2%}")
    
    # DiD analysis
    print("\nDifference-in-Differences Analysis:")
    try:
        did_results = run_did_analysis(df)
        print(did_results.to_string(index=False))
    except Exception as e:
        print(f"Error in DiD analysis: {e}")
    
    # ETI calculation
    eti = calculate_bunching_eti(df)
    print(f"\nETI Lower Bound: {eti:.3f}")
    
    # Subject-level behavior
    print("\nSubject behavior summary:")
    for subject in df['subject_id'].unique():
        subj_df = df[df['subject_id'] == subject]
        
        # Pre-reform average (progressive)
        pre_avg = subj_df[subj_df['round'] <= 8]['labor_supply'].mean()
        
        # Post-reform average (flat25)
        post_avg = subj_df[subj_df['round'] > 8]['labor_supply'].mean()
        
        change = post_avg - pre_avg
        print(f"Subject {subject}: Pre={pre_avg:.1f}, Post={post_avg:.1f}, Change={change:+.1f}")

if __name__ == "__main__":
    main()