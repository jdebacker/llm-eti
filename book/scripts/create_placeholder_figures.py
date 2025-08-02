#!/usr/bin/env python3
"""Create placeholder figures for missing images."""

import matplotlib.pyplot as plt
from pathlib import Path


def create_placeholder_figure(filename, title):
    """Create a placeholder figure."""
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f'Placeholder for:\n{title}', 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=16,
             transform=plt.gca().transAxes)
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Create missing figures
    missing_figures = [
        ("model_eti_comparison.png", "Model ETI Comparison\n(GPT-4o vs GPT-4o-mini)"),
        ("response_rate.png", "Response Rate by Model\n(Percentage of valid responses)")
    ]
    
    for filename, title in missing_figures:
        filepath = figures_dir / filename
        if not filepath.exists():
            create_placeholder_figure(filepath, title)
            print(f"Created placeholder: {filename}")
        else:
            print(f"Already exists: {filename}")


if __name__ == "__main__":
    main()