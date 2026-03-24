"""LLM-ETI: What can LLMs tell us about the ETI?

A research package for investigating how Large Language Models perceive
and simulate behavioral responses to tax policy changes.
"""

__version__ = "0.1.0"

# Import key classes for convenience
from .cache_utils import CacheExplorer
from .config import Config
from .edsl_client import EDSLClient
from .experiment import (
    ExperimentConfig,
    generate_scenarios,
    run_multi_model_experiment,
    run_survey_experiment,
)
from .personas import Persona, create_persona, sample_personas
from .simulation_engine import (
    LabExperimentSimulation,
    SimulationParams,
    TaxSimulation,
)
from .survey import (
    IncomeResponse,
    TaxScenario,
    create_tax_survey_prompt,
    parse_response,
)

# New modules for v2 experimental design
from .tax_brackets import FilingStatus, get_marginal_rate_2024

__all__ = [
    "__version__",
    # Legacy
    "CacheExplorer",
    "Config",
    "EDSLClient",
    "TaxSimulation",
    "LabExperimentSimulation",
    "SimulationParams",
    # v2 - Tax brackets
    "FilingStatus",
    "get_marginal_rate_2024",
    # v2 - Personas
    "Persona",
    "create_persona",
    "sample_personas",
    # v2 - Survey
    "IncomeResponse",
    "TaxScenario",
    "create_tax_survey_prompt",
    "parse_response",
    # v2 - Experiment
    "ExperimentConfig",
    "generate_scenarios",
    "run_survey_experiment",
    "run_multi_model_experiment",
]
