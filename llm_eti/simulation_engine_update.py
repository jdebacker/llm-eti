"""Simulation engine (update) -- reads inputs from PolicyEngine CSV."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .edsl_client_update import EDSLClient


@dataclass
class SimulationParams:
    responses_per_household: int = 1
    test_mode: bool = False


class TaxSimulation:
    """Tax simulation driven by PolicyEngine CSV inputs."""

    def __init__(self, edsl_client: EDSLClient, params: SimulationParams):
        self.client = edsl_client
        self.params = params

    def load_scenarios(self, csv_path: Path) -> pd.DataFrame:
        """Load and validate scenarios from PolicyEngine CSV.

        Filters out rows where broad_income or taxable_income is zero,
        as these produce invalid QuestionNumerical ranges.

        Args:
            csv_path: Path to policyengine_sample_incomes.csv

        Returns:
            Filtered DataFrame of valid scenarios
        """
        df = pd.read_csv(csv_path)

        n_before = len(df)
        df = df[(df["broad_income"] > 0) & (df["taxable_income"] > 0)]
        n_after = len(df)

        if n_before != n_after:
            print(
                f"Filtered {n_before - n_after} zero-income rows ({n_after} remaining)"
            )

        if self.params.test_mode:
            df = df.sample(n=1, random_state=42)
            print("Test mode: sampled 1 row")

        return df.reset_index(drop=True)

    def run_single_simulation(self, row: Dict) -> List[Dict]:
        """Run simulation for a single household scenario.

        Args:
            row: Dict with broad_income, taxable_income, mtr, mtr_prime

        Returns:
            List of result dicts (one per LLM response)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        scenario = {
            "broad_income": row["broad_income"],
            "taxable_income": row["taxable_income"],
            "mtr_last": row["mtr"],
            "mtr_this": row["mtr_prime"],
        }

        try:
            results = self.client.run_batch_surveys(
                [scenario],
                n=self.params.responses_per_household,
                survey_type="tax",
            )

            formatted = []
            for i, result in enumerate(results):
                formatted.append(
                    {
                        "timestamp": timestamp,
                        "household_id": row.get("household_id"),
                        "filing_status": row.get("filing_status"),
                        "broad_income": row["broad_income"],
                        "taxable_income": row["taxable_income"],
                        "mtr": row["mtr"],
                        "mtr_prime": row["mtr_prime"],
                        "response_number": i + 1,
                        "taxable_income_this": result.get("taxable_income_this"),
                        "broad_income_this": result.get("broad_income_this"),
                        "implied_eti_taxable": result.get("implied_eti_taxable"),
                        "implied_eti_broad": result.get("implied_eti_broad"),
                        "model": result.get("model", self.client.model),
                    }
                )
            return formatted

        except Exception as e:
            import traceback

            print(
                f"\nError for household {row.get('household_id')}: {type(e).__name__}: {e}"
            )
            traceback.print_exc()
            return []

    def run_bulk_simulation(self, csv_path: Path) -> pd.DataFrame:
        """Run simulations for all households in the CSV.

        Args:
            csv_path: Path to policyengine_sample_incomes.csv

        Returns:
            DataFrame of all results
        """
        scenarios_df = self.load_scenarios(csv_path)
        all_results = []

        with tqdm(total=len(scenarios_df), desc="Running simulations") as pbar:
            for _, row in scenarios_df.iterrows():
                results = self.run_single_simulation(row.to_dict())
                all_results.extend(results)
                pbar.update(1)

        return pd.DataFrame(all_results)
