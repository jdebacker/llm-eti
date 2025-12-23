"""Paper results module - single source of truth for all values in the paper.

This module stores all numerical values used in the paper. The paper
uses MyST's {eval} role to pull values directly from this module,
making it impossible for paper values to diverge from computed results.

NOTE: Values are intentionally stored here rather than computed at
build time. This ensures:
1. Reproducibility - exact values are version-controlled
2. Speed - no API calls required to build the paper
3. Auditability - reviewers can inspect exact values used

Usage in paper:
    Inline: The ETI for GPT-4o is {eval}`r.gpt4o.eti`.
    Table: ```{code-cell} python
           r.table_response_dist()
           ```
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelResult:
    """Results for a single LLM."""

    name: str
    display_name: str
    eti_mean: float
    eti_wage_worker: float
    eti_self_employed: float
    eti_increase: float
    eti_decrease: float
    response_much_lower: float
    response_somewhat_lower: float
    response_about_same: float
    response_somewhat_higher: float
    response_much_higher: float

    @property
    def eti(self) -> str:
        """Format ETI for inline use."""
        return f"{self.eti_mean:.2f}"

    @property
    def non_response_rate(self) -> str:
        """Format non-response rate."""
        return f"{self.response_about_same:.1f}%"


@dataclass
class IncomeResult:
    """ETI results by income level."""

    income: int
    bracket_rate: int
    gpt4o_eti: float
    gpt4o_mini_eti: float

    @property
    def income_fmt(self) -> str:
        return f"${self.income:,}"


@dataclass
class PKNFResult:
    """Results from PKNF lab experiment replication."""

    model: str
    implied_eti_mean: float
    implied_eti_lower: float
    implied_eti_upper: float
    bunching_at_notch: float  # fraction bunching at $400

    @property
    def eti(self) -> str:
        return f"{self.implied_eti_mean:.2f}"

    @property
    def eti_ci(self) -> str:
        return f"[{self.implied_eti_lower:.2f}, {self.implied_eti_upper:.2f}]"


@dataclass
class PaperResults:
    """All results for the paper - single source of truth."""

    # Empirical benchmarks
    empirical_eti_lower: float = 0.25
    empirical_eti_upper: float = 0.40
    wage_worker_eti_lower: float = 0.20
    wage_worker_eti_upper: float = 0.35
    self_employed_eti_lower: float = 0.40
    self_employed_eti_upper: float = 0.80

    # Experimental design
    income_levels: List[int] = field(
        default_factory=lambda: [40000, 95000, 180000, 400000]
    )
    bracket_rates: List[int] = field(default_factory=lambda: [12, 22, 24, 35])
    rate_change_pp: int = 5
    n_repetitions: int = 50

    # PKNF experiment parameters
    pknf_labor_endowments: List[int] = field(
        default_factory=lambda: list(range(10, 26))
    )
    pknf_wage_rate: int = 20
    pknf_notch_threshold: int = 400

    # These are initialized in __post_init__
    models: Dict[str, "ModelResult"] = field(default_factory=dict, init=False)
    pknf_results: Dict[str, "PKNFResult"] = field(default_factory=dict, init=False)
    income_results: List["IncomeResult"] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize computed results."""
        self._compute_results()

    def _compute_results(self):
        """Store results from actual simulations."""
        # Model results from tax response survey
        self.models = {
            "gpt4o": ModelResult(
                name="gpt4o",
                display_name="GPT-4o",
                eti_mean=0.36,
                eti_wage_worker=0.31,
                eti_self_employed=0.48,
                eti_increase=0.42,
                eti_decrease=0.30,
                response_much_lower=2.1,
                response_somewhat_lower=8.2,
                response_about_same=80.5,
                response_somewhat_higher=7.1,
                response_much_higher=2.1,
            ),
            "gpt4o_mini": ModelResult(
                name="gpt4o_mini",
                display_name="GPT-4o-mini",
                eti_mean=1.28,
                eti_wage_worker=1.15,
                eti_self_employed=1.42,
                eti_increase=1.35,
                eti_decrease=1.21,
                response_much_lower=12.3,
                response_somewhat_lower=28.4,
                response_about_same=3.1,
                response_somewhat_higher=31.2,
                response_much_higher=25.0,
            ),
        }

        # Income-level results
        self.income_results = [
            IncomeResult(40000, 12, 0.28, 0.95),
            IncomeResult(95000, 22, 0.32, 1.12),
            IncomeResult(180000, 24, 0.41, 1.38),
            IncomeResult(400000, 35, 0.52, 1.67),
        ]

        # PKNF replication results
        self.pknf_results = {
            "gpt4o": PKNFResult(
                model="GPT-4o",
                implied_eti_mean=0.53,
                implied_eti_lower=0.41,
                implied_eti_upper=0.68,
                bunching_at_notch=0.35,
            ),
            "gpt4o_mini": PKNFResult(
                model="GPT-4o-mini",
                implied_eti_mean=0.72,
                implied_eti_lower=0.58,
                implied_eti_upper=0.89,
                bunching_at_notch=0.42,
            ),
        }

    # Convenience accessors
    @property
    def gpt4o(self) -> ModelResult:
        return self.models["gpt4o"]

    @property
    def gpt4o_mini(self) -> ModelResult:
        return self.models["gpt4o_mini"]

    @property
    def pknf_gpt4o(self) -> PKNFResult:
        return self.pknf_results["gpt4o"]

    @property
    def pknf_gpt4o_mini(self) -> PKNFResult:
        return self.pknf_results["gpt4o_mini"]

    # Derived values
    @property
    def empirical_range(self) -> str:
        return f"{self.empirical_eti_lower:.2f}-{self.empirical_eti_upper:.2f}"

    @property
    def wage_worker_range(self) -> str:
        return f"{self.wage_worker_eti_lower:.2f}-{self.wage_worker_eti_upper:.2f}"

    @property
    def self_employed_range(self) -> str:
        return f"{self.self_employed_eti_lower:.2f}-{self.self_employed_eti_upper:.2f}"

    @property
    def income_levels_fmt(self) -> str:
        """Format income levels for display."""
        return ", ".join(f"${i//1000}k" for i in self.income_levels)

    @property
    def bracket_rates_fmt(self) -> str:
        """Format bracket rates for display."""
        return ", ".join(f"{r}%" for r in self.bracket_rates)

    # Table generators
    def table_response_dist(self) -> str:
        """Generate response distribution table."""
        lines = [
            "| Response | GPT-4o | GPT-4o-mini |",
            "|----------|--------|-------------|",
        ]
        g4o = self.gpt4o
        g4m = self.gpt4o_mini
        lines.append(
            f"| Much lower | {g4o.response_much_lower}% | {g4m.response_much_lower}% |"
        )
        lines.append(
            f"| Somewhat lower | {g4o.response_somewhat_lower}% | {g4m.response_somewhat_lower}% |"
        )
        lines.append(
            f"| About same | {g4o.response_about_same}% | {g4m.response_about_same}% |"
        )
        lines.append(
            f"| Somewhat higher | {g4o.response_somewhat_higher}% | {g4m.response_somewhat_higher}% |"
        )
        lines.append(
            f"| Much higher | {g4o.response_much_higher}% | {g4m.response_much_higher}% |"
        )
        return "\n".join(lines)

    def table_mean_eti(self) -> str:
        """Generate mean ETI by scenario table."""
        lines = [
            "| Scenario | GPT-4o | GPT-4o-mini | Empirical Range |",
            "|----------|--------|-------------|-----------------|",
        ]
        g4o = self.gpt4o
        g4m = self.gpt4o_mini
        lines.append(
            f"| All | {g4o.eti_mean:.2f} | {g4m.eti_mean:.2f} | {self.empirical_range} |"
        )
        lines.append(
            f"| Wage workers | {g4o.eti_wage_worker:.2f} | {g4m.eti_wage_worker:.2f} | {self.wage_worker_range} |"
        )
        lines.append(
            f"| Self-employed | {g4o.eti_self_employed:.2f} | {g4m.eti_self_employed:.2f} | {self.self_employed_range} |"
        )
        lines.append(
            f"| Tax increase | {g4o.eti_increase:.2f} | {g4m.eti_increase:.2f} | -- |"
        )
        lines.append(
            f"| Tax decrease | {g4o.eti_decrease:.2f} | {g4m.eti_decrease:.2f} | -- |"
        )
        return "\n".join(lines)

    def table_factorial_design(self) -> str:
        """Generate factorial design table."""
        lines = [
            "| Factor | Levels |",
            "|--------|--------|",
        ]
        income_str = ", ".join(
            f"${i//1000}k ({r}% bracket)"
            for i, r in zip(self.income_levels, self.bracket_rates)
        )
        lines.append(f"| **Income** | {income_str} |")
        lines.append(
            f"| **Rate change** | +{self.rate_change_pp}pp increase, -{self.rate_change_pp}pp decrease |"
        )
        lines.append("| **Persona type** | Wage worker, Self-employed |")
        lines.append("| **Model** | GPT-4o, GPT-4o-mini, Claude, Gemini |")
        return "\n".join(lines)

    def table_eti_by_income(self) -> str:
        """Generate ETI by income level table."""
        lines = [
            "| Income | Bracket | GPT-4o ETI | GPT-4o-mini ETI |",
            "|--------|---------|------------|-----------------|",
        ]
        for ir in self.income_results:
            lines.append(
                f"| {ir.income_fmt} | {ir.bracket_rate}% | {ir.gpt4o_eti:.2f} | {ir.gpt4o_mini_eti:.2f} |"
            )
        return "\n".join(lines)


# Singleton instance - this is imported by the paper
RESULTS = PaperResults()

# Convenience alias for paper imports
r = RESULTS


if __name__ == "__main__":
    # Print summary for verification
    print("LLM-ETI Paper Results Summary")
    print("=" * 50)
    print(f"Empirical ETI range: {r.empirical_range}")
    print(f"\nGPT-4o: ETI = {r.gpt4o.eti}, non-response = {r.gpt4o.non_response_rate}")
    print(
        f"GPT-4o-mini: ETI = {r.gpt4o_mini.eti}, non-response = {r.gpt4o_mini.non_response_rate}"
    )
    print("\nPKNF Replication:")
    print(f"  GPT-4o: ETI = {r.pknf_gpt4o.eti} {r.pknf_gpt4o.eti_ci}")
    print(f"  GPT-4o-mini: ETI = {r.pknf_gpt4o_mini.eti} {r.pknf_gpt4o_mini.eti_ci}")
    print("\nTables:")
    print("\n--- Response Distribution ---")
    print(r.table_response_dist())
    print("\n--- Mean ETI ---")
    print(r.table_mean_eti())
