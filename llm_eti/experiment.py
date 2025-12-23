"""
Experiment orchestration for LLM tax surveys.

This module handles:
- Generating factorial scenarios
- Running surveys across models
- Aggregating results
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .analysis import response_to_eti
from .personas import Persona
from .survey import (
    TaxScenario,
    create_tax_survey_prompt,
    parse_response,
)
from .tax_brackets import FilingStatus, get_marginal_rate_2024


@dataclass
class ExperimentConfig:
    """Configuration for a tax response experiment."""

    income_levels: List[float] = field(
        default_factory=lambda: [40000, 95000, 180000, 400000]
    )
    rate_changes: List[float] = field(default_factory=lambda: [0.05, -0.05])
    persona_types: List[str] = field(
        default_factory=lambda: ["wage_worker", "self_employed"]
    )
    n_repetitions: int = 50
    models: List[str] = field(default_factory=lambda: ["gpt-4o-mini"])


def generate_scenarios(
    income_levels: List[float],
    rate_changes: List[float],
    persona_types: List[str],
) -> List[TaxScenario]:
    """
    Generate factorial scenarios for the experiment.

    Args:
        income_levels: List of income levels to test
        rate_changes: List of rate changes (positive = increase)
        persona_types: List of persona type identifiers

    Returns:
        List of TaxScenario objects
    """
    scenarios = []

    for income in income_levels:
        # Get base marginal rate for this income (assume single filer)
        base_rate = get_marginal_rate_2024(income, FilingStatus.SINGLE)

        for rate_change in rate_changes:
            new_rate = base_rate + rate_change
            # Ensure rate stays in valid range
            new_rate = max(0.0, min(0.50, new_rate))

            for persona_type in persona_types:
                # Create persona description based on type
                if persona_type == "wage_worker":
                    description = (
                        f"a 35-year-old employee earning ${income:,.0f} annually"
                    )
                    is_self_employed = False
                elif persona_type == "self_employed":
                    description = f"a 40-year-old self-employed consultant earning ${income:,.0f} annually"
                    is_self_employed = True
                else:
                    description = f"a taxpayer earning ${income:,.0f} annually"
                    is_self_employed = False

                scenario = TaxScenario(
                    persona_description=description,
                    filing_status=FilingStatus.SINGLE,
                    wage_income=income if not is_self_employed else 0,
                    other_income=0 if not is_self_employed else income,
                    current_marginal_rate=base_rate,
                    new_marginal_rate=new_rate,
                )

                scenarios.append(scenario)

    return scenarios


def run_survey_experiment(
    client: Any,  # EDSLClient or mock
    n_scenarios: Optional[int] = None,
    n_repetitions: int = 1,
    config: Optional[ExperimentConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full survey experiment.

    Args:
        client: LLM client (EDSLClient or mock with run_survey method)
        n_scenarios: Number of scenarios (None = use all from config)
        n_repetitions: Responses per scenario
        config: Experiment configuration

    Returns:
        List of result dictionaries
    """
    if config is None:
        config = ExperimentConfig()

    # Generate scenarios
    all_scenarios = generate_scenarios(
        income_levels=config.income_levels,
        rate_changes=config.rate_changes,
        persona_types=config.persona_types,
    )

    if n_scenarios is not None:
        scenarios = all_scenarios[:n_scenarios]
    else:
        scenarios = all_scenarios

    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for scenario in tqdm(scenarios, desc="Running scenarios"):
        prompt = create_tax_survey_prompt(scenario)

        for rep in range(n_repetitions):
            # Run survey
            response = client.run_survey(prompt)

            # Parse response
            if isinstance(response, dict):
                response_text = response.get("response", "")
                explanation = response.get("explanation", "")
            else:
                response_text = str(response)
                explanation = ""

            parsed = parse_response(response_text)

            # Calculate ETI if valid response
            eti = None
            if parsed is not None:
                eti = response_to_eti(
                    response=parsed,
                    current_rate=scenario.current_marginal_rate,
                    new_rate=scenario.new_marginal_rate,
                )

            results.append(
                {
                    "timestamp": timestamp,
                    "persona_description": scenario.persona_description,
                    "filing_status": scenario.filing_status.value,
                    "wage_income": scenario.wage_income,
                    "other_income": scenario.other_income,
                    "total_income": scenario.total_income,
                    "current_rate": scenario.current_marginal_rate,
                    "new_rate": scenario.new_marginal_rate,
                    "rate_change": scenario.rate_change,
                    "is_increase": scenario.is_increase,
                    "repetition": rep + 1,
                    "raw_response": response_text,
                    "parsed_response": parsed.value if parsed else None,
                    "explanation": explanation,
                    "implied_eti": eti,
                }
            )

    return results


def run_multi_model_experiment(
    models: List[str],
    config: Optional[ExperimentConfig] = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    Run experiment across multiple models.

    Args:
        models: List of model names to test
        config: Experiment configuration
        test_mode: If True, use minimal scenarios for testing

    Returns:
        DataFrame with all results
    """
    from .edsl_client import EDSLClient

    if config is None:
        config = ExperimentConfig()

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Running experiment with {model_name}")
        print(f"{'='*60}")

        client = EDSLClient(model=model_name)

        if test_mode:
            n_scenarios = 4
            n_reps = 2
        else:
            n_scenarios = None
            n_reps = config.n_repetitions

        results = run_survey_experiment(
            client=client,
            n_scenarios=n_scenarios,
            n_repetitions=n_reps,
            config=config,
        )

        # Add model to each result
        for r in results:
            r["model"] = model_name

        all_results.extend(results)

    return pd.DataFrame(all_results)


def create_scenario_from_persona(
    persona: Persona,
    rate_change: float,
) -> TaxScenario:
    """
    Create a TaxScenario from a Persona.

    Args:
        persona: Persona object
        rate_change: Rate change in percentage points

    Returns:
        TaxScenario object
    """
    # Get base rate from persona's income
    base_rate = get_marginal_rate_2024(persona.total_income, persona.filing_status)
    new_rate = max(0.0, min(0.50, base_rate + rate_change))

    return TaxScenario(
        persona_description=persona.description,
        filing_status=persona.filing_status,
        wage_income=persona.wage_income,
        other_income=persona.other_income,
        current_marginal_rate=base_rate,
        new_marginal_rate=new_rate,
    )
