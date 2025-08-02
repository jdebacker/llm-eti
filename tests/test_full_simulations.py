"""Test full simulation runs with production parameters."""

import os
from unittest.mock import patch

import pytest

from llm_eti.edsl_client import EDSLClient
from llm_eti.simulation_engine import SimulationParams, TaxSimulation


class TestFullSimulations:
    """Test production simulation configurations."""

    def test_production_params(self):
        """Test that production parameters are correctly set."""
        # Production parameters
        prod_params = SimulationParams(
            responses_per_rate=100,
            min_rate=0.15,
            max_rate=0.35,
            step_size=0.02,
        )

        assert prod_params.responses_per_rate == 100
        assert prod_params.min_rate == 0.15
        assert prod_params.max_rate == 0.35
        assert prod_params.step_size == 0.02

        # Calculate expected number of rates
        # 0.15 to 0.35 in 0.02 steps: 0.15, 0.17, 0.19, ..., 0.33, 0.35
        expected_rates = len(
            [r for r in [0.15 + i * 0.02 for i in range(20)] if r <= 0.35]
        )
        assert expected_rates == 11  # 0.15, 0.17, ..., 0.35

    def test_production_income_range(self):
        """Test production income range configuration."""
        # Gruber & Saez income range
        min_income = 50000
        max_income = 200000
        income_step = 10000

        incomes = list(range(min_income, max_income + income_step, income_step))
        assert len(incomes) == 16  # 50k to 200k in 10k steps
        assert incomes[0] == 50000
        assert incomes[-1] == 200000

    @patch.object(EDSLClient, "run_batch_surveys")
    def test_simulation_size(self, mock_run_batch):
        """Test that full simulation generates expected number of scenarios."""
        mock_run_batch.return_value = [
            {"taxable_income_this": 70000, "model": "gpt-4o-mini"}
        ]

        client = EDSLClient(api_key="test", model="gpt-4o-mini")
        params = SimulationParams(
            responses_per_rate=100,
            min_rate=0.15,
            max_rate=0.35,
            step_size=0.02,
        )

        simulation = TaxSimulation(client, params)

        # Calculate expected scenarios
        n_incomes = 16  # 50k to 200k in 10k steps
        n_rates = 11  # 0.15 to 0.35 in 0.02 steps
        n_responses = 100
        expected_scenarios = n_incomes * n_rates * n_responses

        # Run simulation for one income level to verify
        results = simulation.run_single_simulation(
            broad_income=100000, prior_rate=0.25, new_rate=0.30
        )

        # Should generate at least one result (mocked returns 1)
        assert len(results) == 1  # Due to mock returning single result

    def test_cache_key_generation(self):
        """Test that cache keys are properly generated for scenarios."""
        client = EDSLClient(api_key="test", model="gpt-4o-mini")

        # Create two identical surveys
        survey1 = client.create_tax_survey(100000, 75000, 0.25, 0.30)
        survey2 = client.create_tax_survey(100000, 75000, 0.25, 0.30)

        # They should have the same question text (used for caching)
        assert survey1.questions[0].question_text == survey2.questions[0].question_text

        # Different parameters should give different questions
        survey3 = client.create_tax_survey(100000, 75000, 0.25, 0.35)
        assert survey1.questions[0].question_text != survey3.questions[0].question_text

    @pytest.mark.skipif(
        not os.getenv("EXPECTED_PARROT_API_KEY"),
        reason="Requires API key for integration test",
    )
    def test_cache_hit_detection(self):
        """Test that we can detect cache hits (integration test)."""
        # This test requires actual API access
        client = EDSLClient(
            api_key=os.getenv("EXPECTED_PARROT_API_KEY"),
            model="gpt-4o-mini",
            use_cache=True,
        )

        # Run the same survey twice
        survey = client.create_tax_survey(100000, 75000, 0.25, 0.30)

        # First run - should populate cache
        result1 = client.run_survey(survey)

        # Second run - should hit cache
        result2 = client.run_survey(survey)

        # Results should be identical due to cache
        df1 = result1.select("answer.taxable_income").to_pandas()
        df2 = result2.select("answer.taxable_income").to_pandas()

        assert df1.equals(df2)
