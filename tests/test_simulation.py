import pytest
from simulation_engine import SimulationParams, TaxSimulation
from gpt_utils import GPTClient


class MockGPTClient:
    def get_gpt4_response(self, prompt, n=1):
        return ["75000"] * n

    def parse_income_response(self, response):
        return float(response)

    def calculate_eti(self, *args):
        return 0.4


def test_simulation_params():
    params = SimulationParams(
        min_rate=0.15, max_rate=0.35, step_size=0.02, responses_per_rate=10
    )
    assert params.min_rate == 0.15
    assert params.max_rate == 0.35


def test_simulation_run():
    client = MockGPTClient()
    params = SimulationParams(
        min_rate=0.15, max_rate=0.35, step_size=0.02, responses_per_rate=1
    )

    simulation = TaxSimulation(client, params)
    result = simulation.run_single_simulation(100000, 0.25, 0.30)

    assert result is not None
    assert result["Broad Income"] == 100000
    assert result["Prior Taxable Income"] == 75000
    assert result["Implied ETI"] == 0.4
