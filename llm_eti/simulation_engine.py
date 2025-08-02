"""Simulation engine using EDSL for LLM surveys."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

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
            print(
                f"Error in simulation for income {broad_income}, rate {new_rate}: {str(e)}"
            )
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
    """PKNF lab experiment simulation using EDSL."""

    def __init__(self, edsl_client: EDSLClient):
        self.client = edsl_client

    def run_experiment(
        self, treatments: List[str], rounds: int = 16, subjects_per_treatment: int = 100
    ) -> pd.DataFrame:
        """Run the full lab experiment simulation."""
        all_results = []

        # Treatment mapping
        treatment_map = {
            "Prog,Prog": ["progressive", "progressive"],
            "Prog,Flat25": ["progressive", "flat25"],
            "Prog,Flat50": ["progressive", "flat50"],
            "Flat25,Prog": ["flat25", "progressive"],
            "Flat50,Prog": ["flat50", "progressive"],
        }

        for treatment_name, (first_schedule, second_schedule) in treatment_map.items():
            if treatment_name not in treatments:
                continue

            for subject_id in range(subjects_per_treatment):
                # Random labor endowments for each round
                labor_endowments = np.random.randint(14, 31, size=rounds)

                for round_num in range(rounds):
                    # Determine tax schedule based on round
                    if round_num < 8:
                        schedule = first_schedule
                    else:
                        schedule = second_schedule

                    scenario = {
                        "round_num": round_num + 1,
                        "tax_schedule": schedule,
                        "labor_endowment": int(labor_endowments[round_num]),
                    }

                    # Run survey
                    results = self.client.run_batch_surveys(
                        [scenario], n=1, survey_type="lab"
                    )

                    if results:
                        result = results[0]
                        all_results.append(
                            {
                                "treatment": treatment_name,
                                "subject_id": subject_id,
                                "round_num": round_num + 1,
                                "tax_schedule": schedule,
                                "labor_endowment": labor_endowments[round_num],
                                "labor_supply": result["labor_supply_this"],
                                "income": result["labor_supply_this"]
                                * 20,  # 20 ECU per unit
                                "post_reform": round_num >= 8,
                                "model": result.get("model", self.client.model),
                            }
                        )

        return pd.DataFrame(all_results)
