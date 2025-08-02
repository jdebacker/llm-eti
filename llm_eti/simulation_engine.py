"""Simulation engine using EDSL for LLM surveys."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .edsl_client import EDSLClient


@dataclass
class SimulationParams:
    min_rate: float
    max_rate: float
    step_size: float
    responses_per_rate: int
    taxable_income_ratio: float = 0.75


class TaxSimulation:
    """Tax simulation using EDSL surveys."""

    def __init__(self, edsl_client: EDSLClient, params: SimulationParams):
        self.client = edsl_client
        self.params = params

    def generate_income_ranges(
        self, min_income: float, max_income: float, step: float
    ) -> List[float]:
        return list(np.arange(min_income, max_income + step, step))

    def run_single_simulation(
        self, broad_income: float, prior_rate: float, new_rate: float
    ) -> List[Dict]:
        """Run a single simulation scenario."""
        taxable_income = broad_income * self.params.taxable_income_ratio
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Create scenario for EDSL
            scenario = {
                "broad_income": broad_income,
                "taxable_income": taxable_income,
                "mtr_last": prior_rate,
                "mtr_this": new_rate,
            }

            # Run survey with EDSL
            results = self.client.run_batch_surveys(
                [scenario], n=self.params.responses_per_rate, survey_type="tax"
            )

            # Format results to match existing structure
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(
                    {
                        "timestamp": timestamp,
                        "broad_income": broad_income,
                        "prior_taxable_income": taxable_income,
                        "prior_rate": prior_rate,
                        "new_rate": new_rate,
                        "response_number": i + 1,
                        "raw_response": f"Taxable income: {result['taxable_income_this']}",
                        "parsed_income": result["taxable_income_this"],
                        "implied_eti": result.get("implied_eti"),
                        "model": result.get("model", self.client.model),
                    }
                )

            return formatted_results

        except Exception as e:
            import traceback

            print(f"\nError in simulation for income {broad_income}, rate {new_rate}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            return []

    def run_bulk_simulation(
        self,
        min_income: float,
        max_income: float,
        income_step: float,
        prior_rate: float,
    ) -> pd.DataFrame:
        """Run bulk simulations across income and tax rate ranges."""
        all_results = []
        income_ranges = self.generate_income_ranges(min_income, max_income, income_step)
        tax_rates = np.arange(
            self.params.min_rate,
            self.params.max_rate + self.params.step_size,
            self.params.step_size,
        )

        total_simulations = len(income_ranges) * len(tax_rates)
        with tqdm(total=total_simulations, desc="Running simulations") as pbar:
            for broad_income in income_ranges:
                for rate in tax_rates:
                    results = self.run_single_simulation(
                        broad_income, prior_rate, float(rate)
                    )
                    all_results.extend(results)
                    pbar.update(1)

        return pd.DataFrame(all_results)


# Lab experiment simulation for PKNF replication
class LabExperimentSimulation:
    """PKNF lab experiment simulation using EDSL.

    This class replicates the experimental framework of Pfeil, Kasper, Necker & Feld (2024),
    which measures labor supply responses to changes in tax schedules. The experiment consists of:

    - 16 rounds of decision-making
    - Tax reform after round 8 (either adding or removing a notch)
    - Randomized labor endowments (14-30 units per round)
    - Wage of 20 ECU per unit of labor

    References:
        Pfeil, K., Kasper, M., Necker, S., & Feld, L. P. (2024).
        Tax System Design, Tax Reform, and Labor Supply. CESifo Working Paper No. 11350.
    """

    def __init__(self, edsl_client: EDSLClient):
        self.client = edsl_client
        # Import here to avoid circular imports
        from .config import Config

        self.config = Config.PKNF_CONFIG

    def run_experiment(
        self,
        treatments: List[str],
        rounds: Optional[int] = None,
        subjects_per_treatment: int = 100,
    ) -> pd.DataFrame:
        """Run the full lab experiment simulation.

        Args:
            treatments: List of treatment labels to run (e.g., ["Prog,Prog", "Prog,Flat25"])
            rounds: Number of rounds (default: 16 from config)
            subjects_per_treatment: Number of subjects per treatment group

        Returns:
            DataFrame with experiment results
        """
        from .pknf_types import Treatment

        if rounds is None:
            rounds = int(self.config["rounds"])

        all_results = []

        for treatment_label in treatments:
            try:
                treatment = Treatment.from_label(treatment_label)
            except ValueError:
                print(f"Warning: Unknown treatment '{treatment_label}', skipping")
                continue

            for subject_id in range(subjects_per_treatment):
                # Random labor endowments for each round
                labor_endowments = np.random.randint(
                    int(self.config["labor_endowment_min"]),
                    int(self.config["labor_endowment_max"]) + 1,
                    size=rounds,
                )

                for round_idx in range(rounds):
                    round_num = round_idx + 1  # 1-based round number

                    # Get tax schedule for this round
                    schedule = treatment.get_schedule_for_round(round_num)

                    scenario = {
                        "round_num": round_num,
                        "tax_schedule": schedule.value,
                        "labor_endowment": int(labor_endowments[round_idx]),
                        "wage_per_unit": self.config["wage_per_unit"],
                    }

                    # Run survey
                    results = self.client.run_batch_surveys(
                        [scenario], n=1, survey_type="lab"
                    )

                    if results:
                        result = results[0]
                        labor_supply = result.get("labor_supply_this", 0)

                        all_results.append(
                            {
                                "treatment": treatment.label,
                                "subject_id": subject_id,
                                "round": round_num,  # Changed from round_num to round
                                "tax_schedule": schedule.value,
                                "labor_endowment": labor_endowments[round_idx],
                                "labor_supply": labor_supply,
                                "income": labor_supply * self.config["wage_per_unit"],
                                "post_reform": round_num > self.config["reform_round"],
                                "model": result.get("model", self.client.model),
                            }
                        )

        return pd.DataFrame(all_results)
