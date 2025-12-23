"""
Test suite for tax response survey (TDD - tests written before implementation).

This module tests the experimental design:
- Persona generation from CPS-like demographics
- US tax bracket calculations
- Survey prompt generation
- Response parsing and ETI calculation
"""

import pytest

from llm_eti.analysis import (
    calculate_mean_eti_by_group,
    response_to_eti,
    run_eti_regression,
)
from llm_eti.experiment import generate_scenarios, run_survey_experiment
from llm_eti.personas import create_persona, sample_personas
from llm_eti.survey import (
    IncomeResponse,
    TaxScenario,
    create_tax_survey_prompt,
    parse_response,
)

# Import directly from modules to avoid loading heavy dependencies via __init__
from llm_eti.tax_brackets import FilingStatus, get_marginal_rate_2024

# ==============================================================================
# Tax Bracket Tests
# ==============================================================================


class TestTaxBrackets:
    """Tests for US federal tax bracket calculations."""

    def test_2024_single_brackets(self):
        """Test 2024 single filer tax brackets."""
        # 10% bracket: $0 - $11,600
        assert get_marginal_rate_2024(10000, FilingStatus.SINGLE) == 0.10

        # 12% bracket: $11,601 - $47,150
        assert get_marginal_rate_2024(30000, FilingStatus.SINGLE) == 0.12

        # 22% bracket: $47,151 - $100,525
        assert get_marginal_rate_2024(75000, FilingStatus.SINGLE) == 0.22

        # 24% bracket: $100,526 - $191,950
        assert get_marginal_rate_2024(150000, FilingStatus.SINGLE) == 0.24

        # 32% bracket: $191,951 - $243,725
        assert get_marginal_rate_2024(220000, FilingStatus.SINGLE) == 0.32

        # 35% bracket: $243,726 - $609,350
        assert get_marginal_rate_2024(400000, FilingStatus.SINGLE) == 0.35

        # 37% bracket: $609,351+
        assert get_marginal_rate_2024(700000, FilingStatus.SINGLE) == 0.37

    def test_2024_married_brackets(self):
        """Test 2024 married filing jointly tax brackets."""
        # 12% bracket: $23,201 - $94,300
        assert (
            get_marginal_rate_2024(60000, FilingStatus.MARRIED_FILING_JOINTLY) == 0.12
        )

        # 22% bracket: $94,301 - $201,050
        assert (
            get_marginal_rate_2024(150000, FilingStatus.MARRIED_FILING_JOINTLY) == 0.22
        )

        # 24% bracket: $201,051 - $383,900
        assert (
            get_marginal_rate_2024(300000, FilingStatus.MARRIED_FILING_JOINTLY) == 0.24
        )

    def test_bracket_edge_cases(self):
        """Test bracket boundary conditions."""
        # Exactly at bracket boundary for single
        assert get_marginal_rate_2024(11600, FilingStatus.SINGLE) == 0.10
        assert get_marginal_rate_2024(11601, FilingStatus.SINGLE) == 0.12

    def test_invalid_income_raises(self):
        """Negative income should raise ValueError."""
        with pytest.raises(ValueError, match="Income cannot be negative"):
            get_marginal_rate_2024(-1000, FilingStatus.SINGLE)


# ==============================================================================
# Persona Generation Tests
# ==============================================================================


class TestPersonaGeneration:
    """Tests for creating realistic taxpayer personas."""

    def test_create_persona_basic(self):
        """Test basic persona creation."""
        persona = create_persona(
            name="Alex Chen",
            occupation="Software Engineer",
            filing_status=FilingStatus.SINGLE,
            wage_income=95000,
            other_income=5000,
            num_dependents=0,
            is_self_employed=False,
            age=32,
        )

        assert persona.name == "Alex Chen"
        assert persona.total_income == 100000
        assert persona.filing_status == FilingStatus.SINGLE

    def test_persona_validation_negative_income(self):
        """Personas with negative income should be rejected."""
        with pytest.raises(ValueError, match="Income cannot be negative"):
            create_persona(
                name="Test",
                occupation="Test",
                filing_status=FilingStatus.SINGLE,
                wage_income=-10000,
                other_income=0,
                num_dependents=0,
                is_self_employed=False,
                age=30,
            )

    def test_persona_validation_negative_dependents(self):
        """Personas with negative dependents should be rejected."""
        with pytest.raises(ValueError, match="Dependents cannot be negative"):
            create_persona(
                name="Test",
                occupation="Test",
                filing_status=FilingStatus.SINGLE,
                wage_income=50000,
                other_income=0,
                num_dependents=-1,
                is_self_employed=False,
                age=30,
            )

    def test_sample_personas_from_distribution(self):
        """Test generating personas from income distribution."""
        personas = sample_personas(n=10, seed=42)

        assert len(personas) == 10
        # Should have variety in filing status
        statuses = {p.filing_status for p in personas}
        assert len(statuses) > 1

        # All should have valid incomes
        for p in personas:
            assert p.wage_income >= 0
            assert p.other_income >= 0

    def test_persona_description(self):
        """Test generating natural language persona description."""
        persona = create_persona(
            name="Sarah Johnson",
            occupation="High School Teacher",
            filing_status=FilingStatus.MARRIED_FILING_JOINTLY,
            wage_income=55000,
            other_income=0,
            num_dependents=2,
            is_self_employed=False,
            age=38,
        )

        desc = persona.description
        assert "Sarah Johnson" in desc
        assert "teacher" in desc.lower()
        assert "married" in desc.lower()
        assert "2" in desc  # dependents


# ==============================================================================
# Survey Prompt Tests
# ==============================================================================


class TestSurveyPrompts:
    """Tests for survey prompt generation."""

    def test_prompt_contains_persona_info(self):
        """Prompt should include persona details."""
        scenario = TaxScenario(
            persona_description="a 32-year-old software engineer, single with no dependents",
            filing_status=FilingStatus.SINGLE,
            wage_income=95000,
            other_income=5000,
            current_marginal_rate=0.22,
            new_marginal_rate=0.27,
        )

        prompt = create_tax_survey_prompt(scenario)

        assert "software engineer" in prompt.lower()
        assert "95,000" in prompt or "95000" in prompt
        assert "22%" in prompt
        assert "27%" in prompt

    def test_prompt_includes_response_options(self):
        """Prompt should include all response options."""
        scenario = TaxScenario(
            persona_description="a taxpayer",
            filing_status=FilingStatus.SINGLE,
            wage_income=50000,
            other_income=0,
            current_marginal_rate=0.12,
            new_marginal_rate=0.17,
        )

        prompt = create_tax_survey_prompt(scenario)

        # Response options use underscores in the prompt
        assert "much_lower" in prompt.lower()
        assert "somewhat_lower" in prompt.lower()
        assert "about_same" in prompt.lower()
        assert "somewhat_higher" in prompt.lower()
        assert "much_higher" in prompt.lower()

    def test_prompt_direction_language(self):
        """Prompt should correctly describe increase vs decrease."""
        # Tax increase
        scenario_up = TaxScenario(
            persona_description="a taxpayer",
            filing_status=FilingStatus.SINGLE,
            wage_income=50000,
            other_income=0,
            current_marginal_rate=0.22,
            new_marginal_rate=0.27,
        )
        prompt_up = create_tax_survey_prompt(scenario_up)
        assert "increase" in prompt_up.lower()

        # Tax decrease
        scenario_down = TaxScenario(
            persona_description="a taxpayer",
            filing_status=FilingStatus.SINGLE,
            wage_income=50000,
            other_income=0,
            current_marginal_rate=0.22,
            new_marginal_rate=0.17,
        )
        prompt_down = create_tax_survey_prompt(scenario_down)
        assert "decrease" in prompt_down.lower()


# ==============================================================================
# Response Parsing Tests
# ==============================================================================


class TestResponseParsing:
    """Tests for parsing LLM responses."""

    def test_parse_categorical_response(self):
        """Test parsing categorical responses."""
        assert parse_response("much_lower") == IncomeResponse.MUCH_LOWER
        assert parse_response("MUCH_LOWER") == IncomeResponse.MUCH_LOWER
        assert parse_response("Much Lower") == IncomeResponse.MUCH_LOWER

    def test_parse_response_with_explanation(self):
        """Test parsing response with explanation text."""
        response = """
        My response is: somewhat_lower

        Explanation: With a higher tax rate, I would reduce my overtime hours...
        """
        result = parse_response(response)
        assert result == IncomeResponse.SOMEWHAT_LOWER

    def test_parse_invalid_response(self):
        """Invalid responses should return None."""
        assert parse_response("invalid") is None
        assert parse_response("") is None
        assert parse_response("I don't know") is None


# ==============================================================================
# ETI Calculation Tests
# ==============================================================================


class TestETICalculation:
    """Tests for elasticity of taxable income calculations."""

    def test_eti_from_categorical_response(self):
        """Test converting categorical response to ETI estimate."""
        # Much lower (-15% midpoint) with 5pp rate increase
        # Rate went from 22% to 27%, so net-of-tax went from 78% to 73%
        # % change in (1-t) = (0.73 - 0.78) / 0.78 = -0.0641
        # % change in income = -0.15 (much lower midpoint)
        # ETI = -0.15 / -0.0641 = 2.34
        eti = response_to_eti(
            response=IncomeResponse.MUCH_LOWER,
            current_rate=0.22,
            new_rate=0.27,
        )
        assert 2.0 < eti < 2.7  # Approximately 2.34

    def test_eti_about_same_is_zero(self):
        """About the same response implies zero ETI."""
        eti = response_to_eti(
            response=IncomeResponse.ABOUT_SAME,
            current_rate=0.22,
            new_rate=0.27,
        )
        assert eti == 0.0

    def test_eti_direction_consistency(self):
        """ETI should be positive (income moves with net-of-tax rate)."""
        # Tax increase -> income decrease = positive ETI
        eti_up = response_to_eti(
            response=IncomeResponse.SOMEWHAT_LOWER,
            current_rate=0.22,
            new_rate=0.27,
        )
        assert eti_up > 0

        # Tax decrease -> income increase = positive ETI
        eti_down = response_to_eti(
            response=IncomeResponse.SOMEWHAT_HIGHER,
            current_rate=0.22,
            new_rate=0.17,
        )
        assert eti_down > 0

    def test_eti_no_rate_change(self):
        """No rate change should return None (undefined)."""
        eti = response_to_eti(
            response=IncomeResponse.ABOUT_SAME,
            current_rate=0.22,
            new_rate=0.22,
        )
        assert eti is None


# ==============================================================================
# Experiment Design Tests
# ==============================================================================


class TestExperimentDesign:
    """Tests for factorial experiment design."""

    def test_generate_scenarios(self):
        """Test generating factorial scenarios."""
        scenarios = generate_scenarios(
            income_levels=[40000, 95000, 180000, 400000],
            rate_changes=[0.05, -0.05],
            persona_types=["wage_worker", "self_employed"],
        )

        # 4 income × 2 directions × 2 personas = 16 scenarios
        assert len(scenarios) == 16

        # Check all income levels present (wage_worker has wage_income, self_employed has other_income)
        all_incomes = {
            s.wage_income if s.wage_income > 0 else s.other_income for s in scenarios
        }
        assert all_incomes == {40000, 95000, 180000, 400000}

    def test_scenario_rate_calculation(self):
        """Scenarios should have correct base rates from brackets."""
        scenarios = generate_scenarios(
            income_levels=[40000],  # 12% bracket
            rate_changes=[0.05],
            persona_types=["wage_worker"],
        )

        assert len(scenarios) == 1
        scenario = scenarios[0]
        assert scenario.current_marginal_rate == 0.12
        assert scenario.new_marginal_rate == pytest.approx(0.17)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for the full survey pipeline."""

    @pytest.mark.integration
    def test_full_survey_pipeline_mock(self):
        """Test full pipeline with mocked LLM responses."""
        from unittest.mock import Mock

        # Mock LLM client that returns consistent responses
        mock_client = Mock()
        mock_client.run_survey.return_value = {
            "response": "somewhat_lower",
            "explanation": "Higher taxes mean less incentive to work overtime.",
        }

        results = run_survey_experiment(
            client=mock_client,
            n_scenarios=4,
            n_repetitions=2,
        )

        assert len(results) == 8  # 4 scenarios × 2 repetitions
        assert all("implied_eti" in r for r in results)

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires API key - run manually")
    def test_full_survey_pipeline_real(self):
        """Test with real LLM API (requires key)."""
        from llm_eti.edsl_client import EDSLClient

        client = EDSLClient(model="gpt-4o-mini")
        results = run_survey_experiment(
            client=client,
            n_scenarios=2,
            n_repetitions=1,
        )

        assert len(results) == 2
        for r in results:
            assert r["response"] in [e.value for e in IncomeResponse]


# ==============================================================================
# Regression/Summary Statistics Tests
# ==============================================================================


class TestStatistics:
    """Tests for statistical analysis functions."""

    def test_calculate_mean_eti_by_group(self):
        """Test grouping ETI by persona type."""
        import pandas as pd

        data = pd.DataFrame(
            {
                "persona_type": [
                    "wage_worker",
                    "wage_worker",
                    "self_employed",
                    "self_employed",
                ],
                "implied_eti": [0.3, 0.4, 0.8, 0.9],
            }
        )

        means = calculate_mean_eti_by_group(data, "persona_type")

        assert means["wage_worker"] == pytest.approx(0.35, rel=0.01)
        assert means["self_employed"] == pytest.approx(0.85, rel=0.01)

    def test_regression_with_controls(self):
        """Test regression with demographic controls."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 100
        data = pd.DataFrame(
            {
                "implied_eti": np.random.normal(0.4, 0.2, n),
                "income_100k": np.random.uniform(0.4, 4, n),
                "is_self_employed": np.random.choice([0, 1], n),
                "rate_change": np.random.choice([-0.05, 0.05], n),
            }
        )

        results = run_eti_regression(
            data,
            dependent_var="implied_eti",
            controls=["income_100k", "is_self_employed", "rate_change"],
        )

        assert "coefficients" in results
        assert "std_errors" in results
        assert "r_squared" in results
        assert len(results["coefficients"]) == 4  # intercept + 3 controls
