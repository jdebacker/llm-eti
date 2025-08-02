"""EDSL client for running LLM surveys."""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from edsl import Agent, Jobs, Model, Question, Survey
    from edsl.questions import QuestionNumerical
except ImportError:
    # For testing without EDSL installed
    Question = Survey = Agent = Model = Jobs = None
    QuestionNumerical = None


class EDSLClient:
    """Client for conducting surveys using EDSL."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_cache: bool = True,
    ):
        """Initialize EDSL client.

        Args:
            api_key: Expected Parrot API key. If None, loads from environment.
            model: Model to use for surveys (default: gpt-4o-mini for cost efficiency)
            use_cache: Whether to use EDSL's universal cache (default: True)
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("EXPECTED_PARROT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EXPECTED_PARROT_API_KEY not found in environment or arguments"
            )

        self.model = model
        self.use_cache = use_cache

        # Set API key for EDSL
        if self.api_key:
            os.environ["EXPECTED_PARROT_API_KEY"] = self.api_key

    def create_tax_survey(
        self,
        broad_income: float,
        taxable_income: float,
        mtr_last: float,
        mtr_this: float,
    ) -> "Survey":
        """Create a tax survey for Gruber & Saez replication.

        Args:
            broad_income: Broad income last year
            taxable_income: Taxable income last year
            mtr_last: Marginal tax rate last year (as decimal)
            mtr_this: Marginal tax rate this year (as decimal)

        Returns:
            EDSL Survey object
        """
        # Debug print
        # print(f"DEBUG: broad_income={broad_income}, type={type(broad_income)}")

        # Convert rates to percentages for display
        mtr_last_pct = int(mtr_last * 100)
        mtr_this_pct = int(mtr_this * 100)

        prompt = f"""You are a taxpayer with the following tax profile:
- Your broad income last year was ${broad_income:,.0f}
- Your taxable income last year was ${taxable_income:,.0f}
- Your marginal tax rate last year was {mtr_last_pct}%

This year, if you had the same broad income, your marginal tax rate will change to {mtr_this_pct}%.

Given this change, estimate your taxable income for this year. 
Please provide only a numeric value in dollars."""

        # Ensure numeric types are Python native (not numpy)
        min_val = 0
        max_val = float(broad_income * 2)  # Convert to native Python float

        question = QuestionNumerical(
            question_name="taxable_income",
            question_text=prompt,
            min_value=min_val,
            max_value=max_val,  # Allow for some flexibility
        )

        return Survey([question])

    def create_lab_experiment_survey(
        self,
        round_num: int,
        tax_schedule: str,
        labor_endowment: int,
        wage_per_unit: int = 20,
    ) -> "Survey":
        """Create survey for PKNF lab experiment replication.

        Args:
            round_num: Round number (1-16)
            tax_schedule: Tax schedule type ("flat25", "flat50", "progressive")
            labor_endowment: Maximum labor units available
            wage_per_unit: Wage per unit of labor (default: 20)

        Returns:
            EDSL Survey object
        """
        # Try to use enum for better descriptions
        try:
            from llm_eti.pknf_types import TaxSchedule

            schedule_enum = TaxSchedule(tax_schedule)
            tax_desc = schedule_enum.description
        except (ImportError, ValueError):
            # Fallback to original logic
            if tax_schedule == "flat25":
                tax_desc = "a flat tax rate of 25%"
            elif tax_schedule == "flat50":
                tax_desc = "a flat tax rate of 50%"
            else:  # progressive
                tax_desc = "a progressive tax where income up to 400 is taxed at 25%, and income above 400 is taxed at 50%"

        prompt = f"""You are participating in an economic experiment (Round {round_num}).

You have a labor endowment of {labor_endowment} units.
Each unit of labor you supply earns you {wage_per_unit} experimental currency units (ECU).

The tax system is: {tax_desc}

Note: Under the progressive tax, if you work 20 units you earn 400 ECU and pay 100 ECU in taxes (25%).
If you work 21 units you earn 420 ECU but pay 110 ECU in taxes, leaving you with less after-tax income.

How many units of labor will you supply? (Enter a number between 0 and {labor_endowment})"""

        question = QuestionNumerical(
            question_name="labor_supply",
            question_text=prompt,
            min_value=0,
            max_value=labor_endowment,
        )

        return Survey([question])

    def run_survey(self, survey: "Survey", agent: Optional["Agent"] = None) -> Any:
        """Run a single survey.

        Args:
            survey: EDSL Survey object
            agent: Optional Agent with specific traits

        Returns:
            Survey results
        """
        # Handle model creation with service names for specific providers
        if self.model.startswith("gemini-"):
            model = Model(self.model, service_name="google")
        else:
            model = Model(self.model)

        if agent:
            job = Jobs(survey=survey, agents=[agent], models=[model])
        else:
            job = Jobs(survey=survey, models=[model])

        # Run with caching enabled by default
        results = job.run(cache=self.use_cache)

        return results

    def run_batch_surveys(
        self, scenarios: List[Dict[str, Any]], n: int = 1, survey_type: str = "tax"
    ) -> List[Dict[str, Any]]:
        """Run multiple survey scenarios.

        Args:
            scenarios: List of scenario dictionaries
            n: Number of responses per scenario
            survey_type: Type of survey ("tax" or "lab")

        Returns:
            List of result dictionaries
        """
        all_results = []

        for scenario in scenarios:
            if survey_type == "tax":
                survey = self.create_tax_survey(**scenario)
                result_key = "taxable_income"
            else:  # lab
                survey = self.create_lab_experiment_survey(**scenario)
                result_key = "labor_supply"

            # Create multiple agents for batch processing
            agents = [Agent(name=f"Respondent_{i+1}") for i in range(n)]

            # Handle model creation with service names
            if self.model.startswith("gemini-"):
                model = Model(self.model, service_name="google")
            else:
                model = Model(self.model)

            # Run all agents at once
            job = Jobs(survey=survey, agents=agents, models=[model])
            results = job.run(cache=self.use_cache)

            # Extract results to DataFrame
            df = results.to_pandas()

            # Process each response
            for idx, row in df.iterrows():
                result_dict = scenario.copy()
                result_dict[f"{result_key}_this"] = row[f"answer.{result_key}"]
                result_dict["model"] = row.get("model.model", self.model)

                # Calculate ETI for tax surveys
                if survey_type == "tax":
                    eti = self.calculate_eti(
                        scenario["mtr_last"],
                        scenario["mtr_this"],
                        scenario["taxable_income"],
                        row[f"answer.{result_key}"],
                    )
                    result_dict["implied_eti"] = eti

                all_results.append(result_dict)

        return all_results

    @staticmethod
    def calculate_eti(
        initial_rate: float, new_rate: float, initial_income: float, new_income: float
    ) -> Optional[float]:
        """Calculate elasticity of taxable income.

        Args:
            initial_rate: Initial marginal tax rate
            new_rate: New marginal tax rate
            initial_income: Initial income
            new_income: New income

        Returns:
            ETI value or None if calculation fails
        """
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
