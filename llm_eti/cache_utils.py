"""Utilities for exploring and managing EDSL's universal cache."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from edsl import Cache
except ImportError:
    Cache = None  # Handle case where EDSL isn't installed


class CacheExplorer:
    """Explore and analyze EDSL's universal cache."""

    def __init__(self):
        """Initialize cache explorer."""
        if Cache is None:
            raise ImportError("EDSL is required for cache exploration")
        self.cache = Cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the current cache."""
        try:
            # Get cache entries
            entries = list(self.cache.all())

            stats: Dict[str, Any] = {
                "total_entries": len(entries),
                "models": {},
                "questions": {},
                "total_size_bytes": 0,
            }

            # Analyze entries
            for entry in entries:
                # Count by model
                model = entry.get("model", "unknown")
                stats["models"][model] = stats["models"].get(model, 0) + 1

                # Track question types
                if "question" in entry:
                    q_type = entry["question"].get("type", "unknown")
                    stats["questions"][q_type] = stats["questions"].get(q_type, 0) + 1

            return stats

        except Exception as e:
            return {"error": f"Could not access cache: {str(e)}"}

    def find_tax_scenarios(self, model: Optional[str] = None) -> List[Dict]:
        """Find all tax scenario responses in cache."""
        results = []

        try:
            for entry in self.cache.all():
                # Filter by model if specified
                if model and entry.get("model") != model:
                    continue

                # Look for tax-related questions
                question = entry.get("question", {})
                if "taxable income" in question.get("text", "").lower():
                    results.append(
                        {
                            "model": entry.get("model"),
                            "question": question.get("text"),
                            "answer": entry.get("answer"),
                            "cached_at": entry.get("created_at"),
                        }
                    )

        except Exception:
            pass  # Cache might not be accessible

        return results

    def export_cache_data(self, output_path: Path) -> bool:
        """Export cache data for analysis."""
        try:
            tax_scenarios = self.find_tax_scenarios()

            if tax_scenarios:
                df = pd.DataFrame(tax_scenarios)
                df.to_csv(output_path / "cache_tax_scenarios.csv", index=False)

                # Save summary
                stats = self.get_cache_stats()
                with open(output_path / "cache_stats.json", "w") as f:
                    json.dump(stats, f, indent=2)

                return True

        except Exception as e:
            print(f"Error exporting cache: {e}")

        return False

    def estimate_cost_savings(self, cost_per_1k_tokens: float = 0.0002) -> Dict:
        """Estimate cost savings from cache hits."""
        try:
            entries = list(self.cache.all())

            # Rough estimation: ~100 tokens per tax question/response
            tokens_per_entry = 100
            total_tokens = len(entries) * tokens_per_entry

            savings = {
                "cached_responses": len(entries),
                "estimated_tokens_saved": total_tokens,
                "estimated_cost_saved": (total_tokens / 1000) * cost_per_1k_tokens,
                "cost_per_1k_tokens": cost_per_1k_tokens,
            }

            return savings

        except Exception:
            return {"error": "Could not estimate savings"}


def compare_cached_vs_fresh(client, survey, n: int = 10) -> Dict:
    """Compare cached vs fresh responses for consistency."""
    # Force fresh responses by modifying the question slightly
    import copy

    fresh_survey = copy.deepcopy(survey)
    fresh_survey.questions[0].text += " (fresh run)"

    # Run both
    cached_result = client.run_survey(survey, n=n)
    fresh_result = client.run_survey(fresh_survey, n=n)

    # Convert to DataFrames
    cached_df = cached_result.select("answer.taxable_income").to_pandas()
    fresh_df = fresh_result.select("answer.taxable_income").to_pandas()

    # Calculate statistics
    comparison = {
        "cached_mean": cached_df["answer.taxable_income"].mean(),
        "fresh_mean": fresh_df["answer.taxable_income"].mean(),
        "cached_std": cached_df["answer.taxable_income"].std(),
        "fresh_std": fresh_df["answer.taxable_income"].std(),
        "n_samples": n,
    }

    return comparison
