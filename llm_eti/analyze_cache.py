#!/usr/bin/env python3
"""Command-line tool to analyze EDSL cache usage."""

import argparse
from pathlib import Path

from llm_eti.cache_utils import CacheExplorer


def main():
    """Analyze EDSL cache usage."""
    parser = argparse.ArgumentParser(description="Analyze EDSL cache usage")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export cache data to CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache_analysis"),
        help="Output directory for exports",
    )
    args = parser.parse_args()

    explorer = CacheExplorer()

    # Get and display stats
    print("EDSL Cache Analysis")
    print("=" * 50)

    stats = explorer.get_cache_stats()
    if "error" in stats:
        print(f"Error: {stats['error']}")
        return

    print(f"\nTotal cache entries: {stats['total_entries']}")

    if stats["models"]:
        print("\nEntries by model:")
        for model, count in sorted(stats["models"].items()):
            print(f"  - {model}: {count}")

    if stats["questions"]:
        print("\nEntries by question type:")
        for q_type, count in sorted(stats["questions"].items()):
            print(f"  - {q_type}: {count}")

    # Cost savings
    savings = explorer.estimate_cost_savings()
    if "error" not in savings:
        print("\nCost Savings:")
        print(f"  - Cached responses: {savings['cached_responses']}")
        print(f"  - Estimated tokens saved: {savings['estimated_tokens_saved']:,}")
        print(f"  - Estimated cost saved: ${savings['estimated_cost_saved']:.4f}")
        print(f"  - (at ${savings['cost_per_1k_tokens']} per 1k tokens)")

    # Export if requested
    if args.export:
        args.output_dir.mkdir(exist_ok=True)

        if explorer.export_cache_data(args.output_dir):
            print(f"\nCache data exported to {args.output_dir}/")
        else:
            print("\nNo tax scenarios found to export")

    # Find tax scenarios
    tax_scenarios = explorer.find_tax_scenarios()
    if tax_scenarios:
        print(f"\nFound {len(tax_scenarios)} tax-related scenarios in cache")

        # Show sample
        if len(tax_scenarios) > 0:
            print("\nSample cached question:")
            sample = tax_scenarios[0]
            print(f"  Model: {sample['model']}")
            print(f"  Question: {sample['question'][:100]}...")
            if sample.get("answer"):
                print(f"  Answer: {sample['answer']}")


if __name__ == "__main__":
    main()
