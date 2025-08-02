"""LLM-ETI: What can LLMs tell us about the ETI?

A research package for investigating how Large Language Models perceive
and simulate behavioral responses to tax policy changes.
"""

__version__ = "0.1.0"

# Import key classes for convenience
from .cache_utils import CacheExplorer
from .config import Config
from .edsl_client import EDSLClient
from .simulation_engine import (
    LabExperimentSimulation,
    SimulationParams,
    TaxSimulation,
)

__all__ = [
    "__version__",
    "CacheExplorer",
    "Config",
    "EDSLClient",
    "TaxSimulation",
    "LabExperimentSimulation",
    "SimulationParams",
]
