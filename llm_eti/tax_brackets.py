"""
US Federal Tax Bracket calculations for 2024.

This module provides accurate marginal tax rate lookups based on
actual 2024 IRS tax brackets.
"""

from enum import Enum


class FilingStatus(Enum):
    """Tax filing status."""

    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "married_filing_jointly"
    MARRIED_FILING_SEPARATELY = "married_filing_separately"
    HEAD_OF_HOUSEHOLD = "head_of_household"


# 2024 Federal Tax Brackets
# Source: IRS Revenue Procedure 2023-34
# Format: List of (upper_bound, rate) tuples. Last entry has no upper bound.

BRACKETS_2024 = {
    FilingStatus.SINGLE: [
        (11600, 0.10),
        (47150, 0.12),
        (100525, 0.22),
        (191950, 0.24),
        (243725, 0.32),
        (609350, 0.35),
        (float("inf"), 0.37),
    ],
    FilingStatus.MARRIED_FILING_JOINTLY: [
        (23200, 0.10),
        (94300, 0.12),
        (201050, 0.22),
        (383900, 0.24),
        (487450, 0.32),
        (731200, 0.35),
        (float("inf"), 0.37),
    ],
    FilingStatus.MARRIED_FILING_SEPARATELY: [
        (11600, 0.10),
        (47150, 0.12),
        (100525, 0.22),
        (191950, 0.24),
        (243725, 0.32),
        (365600, 0.35),
        (float("inf"), 0.37),
    ],
    FilingStatus.HEAD_OF_HOUSEHOLD: [
        (16550, 0.10),
        (63100, 0.12),
        (100500, 0.22),
        (191950, 0.24),
        (243700, 0.32),
        (609350, 0.35),
        (float("inf"), 0.37),
    ],
}


def get_marginal_rate_2024(taxable_income: float, filing_status: FilingStatus) -> float:
    """
    Get the marginal tax rate for a given taxable income and filing status.

    Args:
        taxable_income: Taxable income in dollars
        filing_status: Filing status enum

    Returns:
        Marginal tax rate as a decimal (e.g., 0.22 for 22%)

    Raises:
        ValueError: If income is negative
    """
    if taxable_income < 0:
        raise ValueError("Income cannot be negative")

    brackets = BRACKETS_2024[filing_status]

    for upper_bound, rate in brackets:
        if taxable_income <= upper_bound:
            return rate

    # Should never reach here given float("inf") in last bracket
    return brackets[-1][1]


def get_bracket_info(taxable_income: float, filing_status: FilingStatus) -> dict:
    """
    Get detailed bracket information for a given income.

    Args:
        taxable_income: Taxable income in dollars
        filing_status: Filing status enum

    Returns:
        Dictionary with bracket details:
        - marginal_rate: Current marginal rate
        - bracket_floor: Lower bound of current bracket
        - bracket_ceiling: Upper bound of current bracket
        - next_rate: Rate in next higher bracket (None if top)
    """
    if taxable_income < 0:
        raise ValueError("Income cannot be negative")

    brackets = BRACKETS_2024[filing_status]
    prev_ceiling: float = 0.0

    for i, (ceiling, rate) in enumerate(brackets):
        if taxable_income <= ceiling:
            next_rate = brackets[i + 1][1] if i + 1 < len(brackets) else None
            return {
                "marginal_rate": rate,
                "bracket_floor": prev_ceiling,
                "bracket_ceiling": ceiling if ceiling != float("inf") else None,
                "next_rate": next_rate,
            }
        prev_ceiling = ceiling

    # Top bracket
    return {
        "marginal_rate": brackets[-1][1],
        "bracket_floor": brackets[-2][0],
        "bracket_ceiling": None,
        "next_rate": None,
    }


def calculate_tax_liability(
    taxable_income: float, filing_status: FilingStatus
) -> float:
    """
    Calculate total federal income tax liability.

    Args:
        taxable_income: Taxable income in dollars
        filing_status: Filing status enum

    Returns:
        Total tax liability in dollars
    """
    if taxable_income < 0:
        raise ValueError("Income cannot be negative")

    if taxable_income == 0:
        return 0.0

    brackets = BRACKETS_2024[filing_status]
    tax = 0.0
    prev_ceiling: float = 0.0

    for ceiling, rate in brackets:
        if taxable_income <= ceiling:
            # Partial bracket
            tax += (taxable_income - prev_ceiling) * rate
            break
        else:
            # Full bracket
            tax += (ceiling - prev_ceiling) * rate
            prev_ceiling = ceiling

    return tax


def calculate_effective_rate(
    taxable_income: float, filing_status: FilingStatus
) -> float:
    """
    Calculate effective (average) tax rate.

    Args:
        taxable_income: Taxable income in dollars
        filing_status: Filing status enum

    Returns:
        Effective tax rate as a decimal
    """
    if taxable_income <= 0:
        return 0.0

    tax = calculate_tax_liability(taxable_income, filing_status)
    return tax / taxable_income
