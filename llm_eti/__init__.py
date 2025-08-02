"""LLM-ETI: What can LLMs tell us about the ETI?

A research package for investigating how Large Language Models perceive 
and simulate behavioral responses to tax policy changes.
"""

__version__ = "0.1.0"

# Import key classes for convenience
from .config import Config
from .gpt_utils import GPTClient
from .simulation_engine import TaxSimulation, SimulationParams

__all__ = [
    "__version__",
    "Config",
    "GPTClient", 
    "TaxSimulation",
    "SimulationParams",
]