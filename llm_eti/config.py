import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # Base paths
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / "data"
    RESULTS_DIR = ROOT_DIR / "results"

    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Default simulation parameters
    DEFAULT_PARAMS = {
        "min_income": 50000,
        "max_income": 200000,
        "income_step": 10000,
        "min_rate": 0.15,
        "max_rate": 0.35,
        "rate_step": 0.02,
        "responses_per_rate": 100,
        "prior_rate": 0.25,
        "taxable_income_ratio": 0.75,
        "model": "gpt-4o-mini",
    }
