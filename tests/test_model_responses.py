"""Test that models provide expected responses - integration test with real API calls."""

import os
import pytest
from llm_eti.edsl_client import EDSLClient


@pytest.mark.integration
class TestModelResponses:
    """Test actual model responses to ensure they work as expected."""
    
    def test_gpt_provides_numeric_response(self):
        """Test that GPT model provides a valid numeric response."""
        # Skip if no API key
        if not os.getenv("EXPECTED_PARROT_API_KEY"):
            pytest.skip("No EXPECTED_PARROT_API_KEY found")
            
        client = EDSLClient(model="gpt-4o-mini")
        
        # Simple scenario
        scenarios = [{
            "broad_income": 50000,
            "taxable_income": 37500,
            "mtr_last": 0.15,
            "mtr_this": 0.25
        }]
        
        results = client.run_batch_surveys(scenarios, n=1, survey_type="tax")
        
        # Verify we got a result
        assert len(results) == 1
        result = results[0]
        
        # Verify numeric response
        assert "taxable_income_this" in result
        assert isinstance(result["taxable_income_this"], (int, float))
        assert 0 < result["taxable_income_this"] <= 50000  # Should be positive and <= broad income
        
        # Verify ETI calculation
        assert "implied_eti" in result
        assert result["implied_eti"] is not None
        
    def test_model_responds_to_tax_incentive(self):
        """Test that model shows some response to tax rate changes."""
        if not os.getenv("EXPECTED_PARROT_API_KEY"):
            pytest.skip("No EXPECTED_PARROT_API_KEY found")
            
        client = EDSLClient(model="gpt-4o-mini")
        
        # Test both tax increase and decrease
        scenarios = [
            {
                "broad_income": 50000,
                "taxable_income": 37500,
                "mtr_last": 0.15,
                "mtr_this": 0.30  # Tax increase
            },
            {
                "broad_income": 50000,
                "taxable_income": 37500,
                "mtr_last": 0.30,
                "mtr_this": 0.15  # Tax decrease
            }
        ]
        
        results = client.run_batch_surveys(scenarios, n=3, survey_type="tax")
        
        # Get average responses for each scenario
        increase_results = [r for r in results if r["mtr_this"] > r["mtr_last"]]
        decrease_results = [r for r in results if r["mtr_this"] < r["mtr_last"]]
        
        avg_income_increase = sum(r["taxable_income_this"] for r in increase_results) / len(increase_results)
        avg_income_decrease = sum(r["taxable_income_this"] for r in decrease_results) / len(decrease_results)
        
        # When tax increases, taxable income should generally decrease
        # When tax decreases, taxable income should generally increase
        # Allow for some noise but expect the right direction
        assert avg_income_increase < 37500, "Expected lower income when tax rate increases"
        assert avg_income_decrease > 37500, "Expected higher income when tax rate decreases"