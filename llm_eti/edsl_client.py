"""OpenAI client for running tax simulations with two-outcome prompt."""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


class EDSLClient:
    """Client for querying OpenAI models directly."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_cache: bool = True,
    ):
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or arguments")

        self.model = model
        self.use_cache = use_cache
        self.client = OpenAI(api_key=self.api_key)

    def build_prompt(
        self,
        broad_income: float,
        taxable_income: float,
        mtr_last: float,
        mtr_this: float,
    ) -> str:
        mtr_last_pct = int(mtr_last * 100)
        mtr_this_pct = int(mtr_this * 100)

        return f"""You are a taxpayer with the following profile:
- Last year, your broad income was ${broad_income:,.0f}
- Last year, your taxable income was ${taxable_income:,.0f}
- Last year, your marginal tax rate was {mtr_last_pct}%

Due to a change in tax law, your marginal tax rate this year will be {mtr_this_pct}%.
Your broad income before any adjustments would be approximately the same as last year.

Given this change in tax rates, you may adjust your behavior -- for example,
how much you work, your charitable contributions, retirement savings, or the
timing of income realizations like capital gains. What would your taxable
income be this year? And what would your broad income be?

Respond with exactly two lines:
TAXABLE_INCOME: <number>
BROAD_INCOME: <number>"""

    def query(self, prompt: str) -> Optional[Dict[str, float]]:
        """Send prompt to OpenAI and parse the two numeric responses.

        Returns:
            Dict with 'taxable_income' and 'broad_income', or None on failure.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
        )
        text = response.choices[0].message.content.strip()

        taxable = None
        broad = None
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("TAXABLE_INCOME:"):
                try:
                    taxable = float(line.split(":", 1)[1].strip().replace(",", ""))
                except ValueError:
                    pass
            elif line.upper().startswith("BROAD_INCOME:"):
                try:
                    broad = float(line.split(":", 1)[1].strip().replace(",", ""))
                except ValueError:
                    pass

        if taxable is None or broad is None:
            print(f"  Warning: could not parse response:\n{text}")
            return None

        return {"taxable_income": taxable, "broad_income": broad}

    def run_batch_surveys(
        self, scenarios: List[Dict], n: int = 1, survey_type: str = "tax"
    ) -> List[Dict]:
        """Run n queries for each scenario and return results.

        Args:
            scenarios: List of scenario dicts (broad_income, taxable_income, mtr_last, mtr_this)
            n: Number of LLM responses per scenario
            survey_type: Only "tax" is supported in this client

        Returns:
            List of result dicts
        """
        all_results = []

        for scenario in scenarios:
            prompt = self.build_prompt(
                broad_income=scenario["broad_income"],
                taxable_income=scenario["taxable_income"],
                mtr_last=scenario["mtr_last"],
                mtr_this=scenario["mtr_this"],
            )

            for _ in range(n):
                parsed = self.query(prompt)
                if parsed is None:
                    continue

                result = scenario.copy()
                result["taxable_income_this"] = parsed["taxable_income"]
                result["broad_income_this"] = parsed["broad_income"]
                result["model"] = self.model

                result["implied_eti_taxable"] = self.calculate_eti(
                    scenario["mtr_last"],
                    scenario["mtr_this"],
                    scenario["taxable_income"],
                    parsed["taxable_income"],
                )
                result["implied_eti_broad"] = self.calculate_eti(
                    scenario["mtr_last"],
                    scenario["mtr_this"],
                    scenario["broad_income"],
                    parsed["broad_income"],
                )

                all_results.append(result)

        return all_results

    @staticmethod
    def calculate_eti(
        initial_rate: float, new_rate: float, initial_income: float, new_income: float
    ) -> Optional[float]:
        try:
            percent_change_income = (new_income - initial_income) / initial_income
            percent_change_net_of_tax_rate = ((1 - new_rate) - (1 - initial_rate)) / (
                1 - initial_rate
            )
            if percent_change_net_of_tax_rate == 0:
                return None
            return percent_change_income / percent_change_net_of_tax_rate
        except (ZeroDivisionError, TypeError):
            return None
