from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class SimulationParams:
    min_rate: float
    max_rate: float
    step_size: float
    responses_per_rate: int
    taxable_income_ratio: float = 0.75


class TaxSimulation:
    def __init__(self, gpt_client, params: SimulationParams):
        self.gpt_client = gpt_client
        self.params = params

    def generate_income_ranges(
        self, min_income: float, max_income: float, step: float
    ) -> List[float]:
        return np.arange(min_income, max_income + step, step)

    def create_prompt(
        self,
        broad_income: float,
        taxable_income: float,
        prior_rate: float,
        new_rate: float,
    ) -> str:
        return f"""
        You are a taxpayer with the following tax profile:
        - Your broad income last year was ${broad_income:,.2f}
        - Your taxable income last year (after deductions) was ${taxable_income:,.2f}
        - Your marginal tax rate last year was {prior_rate:.2%}
        
        This year, if you had the same broad income as last year, your marginal tax rate will change to {new_rate:.2%}. Given this change, estimate your taxable income for this year.
        Consider how this change in tax rate might affect your behavior, such as your work hours, investment decisions, or tax planning strategies.
        
        Provide your response as a single number representing your estimated taxable income for this year, rounded to the nearest dollar. Do not include any explanations or additional text.
        """

    def run_single_simulation(
        self, broad_income: float, prior_rate: float, new_rate: float
    ) -> List[Dict]:
        taxable_income = broad_income * self.params.taxable_income_ratio
        prompt = self.create_prompt(broad_income, taxable_income, prior_rate, new_rate)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            responses = self.gpt_client.get_gpt4_response(
                prompt, n=self.params.responses_per_rate
            )
            results = []

            for i, response in enumerate(responses):
                parsed_income = self.gpt_client.parse_income_response(response)
                if parsed_income is not None:
                    eti = None
                    if prior_rate != new_rate:  # Avoid division by zero
                        eti = self.gpt_client.calculate_eti(
                            prior_rate, new_rate, taxable_income, parsed_income
                        )

                    results.append(
                        {
                            "timestamp": timestamp,
                            "broad_income": broad_income,
                            "prior_taxable_income": taxable_income,
                            "prior_rate": prior_rate,
                            "new_rate": new_rate,
                            "response_number": i + 1,
                            "raw_response": response,
                            "parsed_income": parsed_income,
                            "implied_eti": eti,
                        }
                    )

            return results
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
                    results = self.run_single_simulation(broad_income, prior_rate, rate)
                    all_results.extend(results)
                    pbar.update(1)

        return pd.DataFrame(all_results)
