"""
Tax survey prompt generation and response parsing.

This module handles creating survey prompts and parsing LLM responses
for the tax response experiment.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .tax_brackets import FilingStatus


class IncomeResponse(Enum):
    """Categorical income response options."""

    MUCH_LOWER = "much_lower"  # down 10%+
    SOMEWHAT_LOWER = "somewhat_lower"  # down 2-10%
    ABOUT_SAME = "about_same"  # within 2%
    SOMEWHAT_HIGHER = "somewhat_higher"  # up 2-10%
    MUCH_HIGHER = "much_higher"  # up 10%+


# Midpoint estimates for each response category
RESPONSE_MIDPOINTS = {
    IncomeResponse.MUCH_LOWER: -0.15,  # -15% (range: -10% to -inf)
    IncomeResponse.SOMEWHAT_LOWER: -0.06,  # -6% (range: -2% to -10%)
    IncomeResponse.ABOUT_SAME: 0.0,  # 0% (range: -2% to +2%)
    IncomeResponse.SOMEWHAT_HIGHER: 0.06,  # +6% (range: +2% to +10%)
    IncomeResponse.MUCH_HIGHER: 0.15,  # +15% (range: +10% to +inf)
}


@dataclass
class TaxScenario:
    """A tax scenario for the survey."""

    persona_description: str
    filing_status: FilingStatus
    wage_income: float
    other_income: float
    current_marginal_rate: float
    new_marginal_rate: float

    @property
    def total_income(self) -> float:
        return self.wage_income + self.other_income

    @property
    def rate_change(self) -> float:
        """Rate change in percentage points (positive = increase)."""
        return self.new_marginal_rate - self.current_marginal_rate

    @property
    def is_increase(self) -> bool:
        """Whether this is a tax increase."""
        return self.rate_change > 0


def create_tax_survey_prompt(scenario: TaxScenario) -> str:
    """
    Create a survey prompt for a tax scenario.

    Args:
        scenario: TaxScenario with all required information

    Returns:
        Formatted prompt string
    """
    # Direction language
    if scenario.is_increase:
        direction_verb = "will increase"
    else:
        direction_verb = "will decrease"

    # Format rates as percentages
    current_pct = int(scenario.current_marginal_rate * 100)
    new_pct = int(scenario.new_marginal_rate * 100)
    change_pct = abs(int(scenario.rate_change * 100))

    # Format income
    wage_str = f"${scenario.wage_income:,.0f}"
    other_str = f"${scenario.other_income:,.0f}" if scenario.other_income > 0 else None

    # Build prompt
    prompt = f"""You are {scenario.persona_description}.

Your current tax situation:
- Filing status: {scenario.filing_status.value.replace('_', ' ')}
- Annual wage/salary income: {wage_str}"""

    if other_str:
        prompt += f"""
- Other income (investments, etc.): {other_str}"""

    prompt += f"""
- Current federal marginal tax rate: {current_pct}%

A tax law change {direction_verb} your marginal tax rate by {change_pct} percentage points, from {current_pct}% to {new_pct}%.

Consider how this might affect your:
1. Work effort (overtime, side jobs, career advancement)
2. Tax planning (timing of income, retirement contributions, deductions)
3. Other financial decisions

Question: Compared to this year, what would your taxable income be NEXT year after the tax change takes effect?

Please select ONE of the following:
- MUCH_LOWER: My taxable income would decrease by 10% or more
- SOMEWHAT_LOWER: My taxable income would decrease by 2-10%
- ABOUT_SAME: My taxable income would stay about the same (within 2%)
- SOMEWHAT_HIGHER: My taxable income would increase by 2-10%
- MUCH_HIGHER: My taxable income would increase by 10% or more

After selecting your response, briefly explain your reasoning.

Your response:"""

    return prompt


def parse_response(response_text: str) -> Optional[IncomeResponse]:
    """
    Parse an LLM response to extract the categorical answer.

    Args:
        response_text: Raw response text from LLM

    Returns:
        IncomeResponse enum value, or None if parsing fails
    """
    if not response_text:
        return None

    # Normalize text
    text = response_text.upper().strip()

    # Try exact matches first
    for response in IncomeResponse:
        if response.value.upper() in text:
            return response

    # Try partial matches
    patterns = {
        IncomeResponse.MUCH_LOWER: [r"MUCH\s*LOWER", r"DECREASE.*10%", r"DOWN.*10%"],
        IncomeResponse.SOMEWHAT_LOWER: [
            r"SOMEWHAT\s*LOWER",
            r"SLIGHTLY\s*LOWER",
            r"DECREASE.*2-10%",
        ],
        IncomeResponse.ABOUT_SAME: [
            r"ABOUT\s*(?:THE\s*)?SAME",
            r"STAY.*SAME",
            r"NO\s*CHANGE",
            r"UNCHANGED",
        ],
        IncomeResponse.SOMEWHAT_HIGHER: [
            r"SOMEWHAT\s*HIGHER",
            r"SLIGHTLY\s*HIGHER",
            r"INCREASE.*2-10%",
        ],
        IncomeResponse.MUCH_HIGHER: [r"MUCH\s*HIGHER", r"INCREASE.*10%", r"UP.*10%"],
    }

    for response, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, text):
                return response

    return None


def create_pknf_lab_prompt(
    round_num: int,
    labor_endowment: int,
    tax_schedule: str,
    wage_per_unit: int = 20,
) -> str:
    """
    Create prompt for PKNF lab experiment replication.

    This mirrors the instructions given to human subjects in PKNF (2024).

    Args:
        round_num: Round number (1-16)
        labor_endowment: Maximum labor units available this round
        tax_schedule: Tax schedule ("flat25", "flat50", "progressive")
        wage_per_unit: Wage per unit of labor

    Returns:
        Formatted prompt string
    """
    # Tax schedule descriptions
    if tax_schedule == "flat25":
        tax_desc = "All of your income is taxed at a flat rate of 25%."
        example = f"If you work all {labor_endowment} hours and earn ${labor_endowment * wage_per_unit}, you pay ${int(labor_endowment * wage_per_unit * 0.25)} in taxes and keep ${int(labor_endowment * wage_per_unit * 0.75)}."
    elif tax_schedule == "flat50":
        tax_desc = "All of your income is taxed at a flat rate of 50%."
        example = f"If you work all {labor_endowment} hours and earn ${labor_endowment * wage_per_unit}, you pay ${int(labor_endowment * wage_per_unit * 0.50)} in taxes and keep ${int(labor_endowment * wage_per_unit * 0.50)}."
    else:  # progressive
        tax_desc = """Income up to $400 is taxed at 25%.
Income above $400 is taxed at 50%."""
        # Example showing the notch
        example = """Working 20 hours = $400 income. Tax = $100 (25%). You keep $300.
Working 21 hours = $420 income. Tax = $100 + $10 (50% on $20 above $400) = $110. You keep $310.
Notice: Working 1 extra hour only nets you $10 after taxes because of the higher rate."""

    prompt = f"""LABOR DECISION - Round {round_num}

You have {labor_endowment} hours available to work this round.
Each hour of work earns ${wage_per_unit}.

TAX SYSTEM:
{tax_desc}

EXAMPLES:
{example}

You can choose to work anywhere from 0 to {labor_endowment} hours.
Your goal is to maximize your after-tax income considering both earnings and leisure.

How many hours will you work? (Enter a number from 0 to {labor_endowment})

Your decision:"""

    return prompt
