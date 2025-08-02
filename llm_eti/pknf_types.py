"""Type definitions for PKNF experiment."""

from enum import Enum


class TaxSchedule(Enum):
    """Tax schedule types in the PKNF experiment."""

    FLAT_25 = "flat25"
    FLAT_50 = "flat50"
    PROGRESSIVE = "progressive"

    @property
    def description(self) -> str:
        """Get human-readable description of the tax schedule."""
        if self == TaxSchedule.FLAT_25:
            return "a flat tax rate of 25%"
        elif self == TaxSchedule.FLAT_50:
            return "a flat tax rate of 50%"
        else:  # PROGRESSIVE
            return "a progressive tax where income up to 400 is taxed at 25%, and income above 400 is taxed at 50%"


class Treatment(Enum):
    """Treatment types representing tax system transitions.

    Each treatment consists of:
    - A label (used in data/results)
    - Pre-reform tax schedule (rounds 1-8)
    - Post-reform tax schedule (rounds 9-16)
    """

    # Control group: no change
    CONTROL = ("Prog,Prog", TaxSchedule.PROGRESSIVE, TaxSchedule.PROGRESSIVE)

    # Treatment groups: progressive to flat
    PROG_TO_FLAT25 = ("Prog,Flat25", TaxSchedule.PROGRESSIVE, TaxSchedule.FLAT_25)
    PROG_TO_FLAT50 = ("Prog,Flat50", TaxSchedule.PROGRESSIVE, TaxSchedule.FLAT_50)

    # Treatment groups: flat to progressive
    FLAT25_TO_PROG = ("Flat25,Prog", TaxSchedule.FLAT_25, TaxSchedule.PROGRESSIVE)
    FLAT50_TO_PROG = ("Flat50,Prog", TaxSchedule.FLAT_50, TaxSchedule.PROGRESSIVE)

    def __init__(self, label: str, pre_reform: TaxSchedule, post_reform: TaxSchedule):
        self.label = label
        self.pre_reform = pre_reform
        self.post_reform = post_reform

    @classmethod
    def from_label(cls, label: str) -> "Treatment":
        """Get treatment from its string label."""
        for treatment in cls:
            if treatment.label == label:
                return treatment
        raise ValueError(f"Unknown treatment label: {label}")

    def get_schedule_for_round(self, round_num: int) -> TaxSchedule:
        """Get the tax schedule for a given round (1-based)."""
        if round_num < 1 or round_num > 16:
            raise ValueError(f"Round number must be between 1 and 16, got {round_num}")

        # Rounds 1-8: pre-reform, Rounds 9-16: post-reform
        if round_num <= 8:
            return self.pre_reform
        else:
            return self.post_reform

    @property
    def introduces_notch(self) -> bool:
        """Check if this treatment introduces a progressive tax notch."""
        return (
            self.pre_reform in [TaxSchedule.FLAT_25, TaxSchedule.FLAT_50]
            and self.post_reform == TaxSchedule.PROGRESSIVE
        )

    @property
    def removes_notch(self) -> bool:
        """Check if this treatment removes a progressive tax notch."""
        return self.pre_reform == TaxSchedule.PROGRESSIVE and self.post_reform in [
            TaxSchedule.FLAT_25,
            TaxSchedule.FLAT_50,
        ]
