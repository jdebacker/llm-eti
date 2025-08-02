"""Test EDSL integration for LLM ETI surveys."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd


class TestEDSLIntegration:
    """Test the EDSL survey implementation."""

    def test_edsl_client_initialization(self):
        """Test that EDSLClient initializes properly."""
        from llm_eti.edsl_client import EDSLClient

        client = EDSLClient(api_key="test_key", model="gpt-4o-mini")
        assert client.api_key == "test_key"
        assert client.model == "gpt-4o-mini"

    def test_create_tax_survey(self):
        """Test creating a tax survey with EDSL."""
        from llm_eti.edsl_client import EDSLClient

        client = EDSLClient(api_key="test_key")
        survey = client.create_tax_survey(
            broad_income=100000, taxable_income=75000, mtr_last=0.25, mtr_this=0.30
        )

        # Check survey has the right question
        assert len(survey.questions) == 1
        assert "taxable income" in survey.questions[0].text.lower()

    def test_run_survey_batch(self):
        """Test running a batch of surveys."""
        from llm_eti.edsl_client import EDSLClient

        client = EDSLClient(api_key="test_key", model="gpt-4o-mini")

        # Mock the survey run
        with patch.object(client, "run_survey") as mock_run:
            mock_result = MagicMock()
            mock_result.select.return_value.to_pandas.return_value = pd.DataFrame(
                {"answer.taxable_income": [72000], "model": ["gpt-4o-mini"]}
            )
            mock_run.return_value = mock_result

            scenarios = [
                {
                    "broad_income": 100000,
                    "taxable_income": 75000,
                    "mtr_last": 0.25,
                    "mtr_this": 0.30,
                }
            ]

            results = client.run_batch_surveys(scenarios, n=1)

            assert len(results) == 1
            assert results[0]["taxable_income_this"] == 72000
            assert results[0]["model"] == "gpt-4o-mini"

    def test_calculate_eti(self):
        """Test ETI calculation."""
        from llm_eti.edsl_client import EDSLClient

        eti = EDSLClient.calculate_eti(
            initial_rate=0.25, new_rate=0.30, initial_income=75000, new_income=72000
        )

        # ETI = (% change in income) / (% change in 1-MTR)
        # % change income = (72000 - 75000) / 75000 = -0.04
        # % change 1-MTR = ((1-0.30) - (1-0.25)) / (1-0.25) = -0.0667
        # ETI = -0.04 / -0.0667 ≈ 0.6
        assert abs(eti - 0.6) < 0.01

    def test_uses_cache(self):
        """Test that EDSL uses caching for repeated queries."""
        from llm_eti.edsl_client import EDSLClient

        client = EDSLClient(api_key="test_key", model="gpt-4o-mini", use_cache=True)

        # Mock survey execution
        with patch("edsl.jobs.Jobs.run") as mock_run:
            mock_result = MagicMock()
            mock_result.select.return_value.to_pandas.return_value = pd.DataFrame(
                {"answer.taxable_income": [72000], "model": ["gpt-4o-mini"]}
            )
            mock_run.return_value = mock_result

            # Run same survey twice
            survey = client.create_tax_survey(100000, 75000, 0.25, 0.30)
            client.run_survey(survey)
            client.run_survey(survey)

            # Should only call API once due to caching
            assert mock_run.call_count == 1

    def test_environment_variable_loading(self):
        """Test loading API key from environment."""
        os.environ["EXPECTED_PARROT_API_KEY"] = "test_env_key"

        from llm_eti.edsl_client import EDSLClient

        client = EDSLClient()
        assert client.api_key == "test_env_key"

        # Clean up
        del os.environ["EXPECTED_PARROT_API_KEY"]

    def test_lab_experiment_prompt(self):
        """Test PKNF lab experiment prompt generation."""
        from llm_eti.edsl_client import EDSLClient

        client = EDSLClient(api_key="test_key")
        survey = client.create_lab_experiment_survey(
            round_num=1,
            tax_schedule="progressive",
            labor_endowment=25,
            wage_per_unit=20,
        )

        question = survey.questions[0]
        assert "labor endowment" in question.text
        assert "25" in question.text
        assert "tax" in question.text.lower()
