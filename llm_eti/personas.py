"""
Taxpayer persona generation for LLM tax surveys.

This module creates realistic taxpayer personas based on
demographic distributions similar to CPS/ACS microdata.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .tax_brackets import FilingStatus


@dataclass
class Persona:
    """A taxpayer persona for the survey."""

    name: str
    occupation: str
    filing_status: FilingStatus
    wage_income: float
    other_income: float
    num_dependents: int
    is_self_employed: bool
    age: int

    @property
    def total_income(self) -> float:
        """Total income from all sources."""
        return self.wage_income + self.other_income

    @property
    def description(self) -> str:
        """Generate a natural language description of this persona."""
        # Filing status description
        if self.filing_status == FilingStatus.SINGLE:
            status_desc = "single"
        elif self.filing_status == FilingStatus.MARRIED_FILING_JOINTLY:
            status_desc = "married"
        elif self.filing_status == FilingStatus.HEAD_OF_HOUSEHOLD:
            status_desc = "single parent"
        else:
            status_desc = "married, filing separately"

        # Dependents description
        if self.num_dependents == 0:
            dep_desc = "no dependents"
        elif self.num_dependents == 1:
            dep_desc = "1 dependent"
        else:
            dep_desc = f"{self.num_dependents} dependents"

        # Employment description
        if self.is_self_employed:
            emp_desc = f"self-employed {self.occupation.lower()}"
        else:
            emp_desc = self.occupation.lower()

        return (
            f"{self.name}, a {self.age}-year-old {emp_desc}, "
            f"{status_desc} with {dep_desc}"
        )


def create_persona(
    name: str,
    occupation: str,
    filing_status: FilingStatus,
    wage_income: float,
    other_income: float,
    num_dependents: int,
    is_self_employed: bool,
    age: int,
) -> Persona:
    """
    Create a validated persona.

    Args:
        name: Person's name
        occupation: Job title/occupation
        filing_status: Tax filing status
        wage_income: Wage/salary income
        other_income: Other income (investments, etc.)
        num_dependents: Number of dependents
        is_self_employed: Whether self-employed
        age: Age in years

    Returns:
        Validated Persona object

    Raises:
        ValueError: If any validation fails
    """
    if wage_income < 0:
        raise ValueError("Income cannot be negative")
    if other_income < 0:
        raise ValueError("Income cannot be negative")
    if num_dependents < 0:
        raise ValueError("Dependents cannot be negative")
    if age < 0 or age > 120:
        raise ValueError("Age must be between 0 and 120")

    return Persona(
        name=name,
        occupation=occupation,
        filing_status=filing_status,
        wage_income=wage_income,
        other_income=other_income,
        num_dependents=num_dependents,
        is_self_employed=is_self_employed,
        age=age,
    )


# Predefined persona templates for factorial design
PERSONA_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "Alex Chen",
        "occupation": "Software Engineer",
        "filing_status": FilingStatus.SINGLE,
        "num_dependents": 0,
        "is_self_employed": False,
        "age": 32,
    },
    {
        "name": "Sarah Johnson",
        "occupation": "High School Teacher",
        "filing_status": FilingStatus.MARRIED_FILING_JOINTLY,
        "num_dependents": 2,
        "is_self_employed": False,
        "age": 38,
    },
    {
        "name": "Marcus Williams",
        "occupation": "Freelance Consultant",
        "filing_status": FilingStatus.SINGLE,
        "num_dependents": 0,
        "is_self_employed": True,
        "age": 45,
    },
    {
        "name": "Patricia Miller",
        "occupation": "Retired Accountant",
        "filing_status": FilingStatus.MARRIED_FILING_JOINTLY,
        "num_dependents": 0,
        "is_self_employed": False,
        "age": 68,
    },
]

# Income distributions by occupation type (rough CPS-based)
INCOME_DISTRIBUTIONS = {
    "Software Engineer": {"mean": 120000, "std": 40000, "min": 60000, "max": 300000},
    "High School Teacher": {"mean": 60000, "std": 15000, "min": 40000, "max": 100000},
    "Freelance Consultant": {"mean": 90000, "std": 50000, "min": 30000, "max": 250000},
    "Retired Accountant": {"mean": 50000, "std": 20000, "min": 20000, "max": 150000},
}

# Names for random generation
FIRST_NAMES = [
    "James",
    "Mary",
    "Michael",
    "Patricia",
    "Robert",
    "Jennifer",
    "David",
    "Linda",
    "William",
    "Elizabeth",
    "Richard",
    "Barbara",
    "Joseph",
    "Susan",
    "Thomas",
    "Jessica",
    "Charles",
    "Sarah",
    "Christopher",
    "Karen",
    "Daniel",
    "Lisa",
    "Matthew",
    "Nancy",
    "Anthony",
    "Betty",
    "Mark",
    "Margaret",
    "Donald",
    "Sandra",
    "Wei",
    "Priya",
    "Mohammed",
    "Fatima",
    "Carlos",
    "Maria",
    "Hiroshi",
    "Yuki",
    "Ahmed",
    "Aisha",
    "Ivan",
    "Olga",
]

LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Chen",
    "Patel",
    "Kim",
    "Nguyen",
    "Singh",
    "Ali",
    "Yamamoto",
    "Ivanov",
    "Mueller",
    "Andersson",
    "Johansson",
]

OCCUPATIONS = [
    "Software Engineer",
    "Teacher",
    "Nurse",
    "Accountant",
    "Sales Manager",
    "Marketing Specialist",
    "Financial Analyst",
    "Lawyer",
    "Doctor",
    "Retail Manager",
    "Administrative Assistant",
    "Construction Worker",
    "Electrician",
    "Real Estate Agent",
    "Consultant",
]


def sample_personas(n: int, seed: Optional[int] = None) -> List[Persona]:
    """
    Generate n random personas from realistic distributions.

    Args:
        n: Number of personas to generate
        seed: Random seed for reproducibility

    Returns:
        List of Persona objects
    """
    if seed is not None:
        random.seed(seed)

    personas = []

    for _ in range(n):
        # Random name
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"

        # Random occupation
        occupation = random.choice(OCCUPATIONS)

        # Random filing status (weighted by US distribution)
        # Approx: 40% single, 45% married joint, 10% HoH, 5% married separate
        status_roll = random.random()
        if status_roll < 0.40:
            filing_status = FilingStatus.SINGLE
            num_dependents = 0
        elif status_roll < 0.85:
            filing_status = FilingStatus.MARRIED_FILING_JOINTLY
            num_dependents = random.choices(
                [0, 1, 2, 3], weights=[0.3, 0.25, 0.3, 0.15]
            )[0]
        elif status_roll < 0.95:
            filing_status = FilingStatus.HEAD_OF_HOUSEHOLD
            num_dependents = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        else:
            filing_status = FilingStatus.MARRIED_FILING_SEPARATELY
            num_dependents = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]

        # Self-employed (about 10% of workforce)
        is_self_employed = random.random() < 0.10

        # Age (working age distribution)
        age = int(random.gauss(42, 12))
        age = max(22, min(70, age))

        # Income based on occupation and age
        base_income = random.gauss(75000, 40000)
        # Age premium (peaks around 50)
        age_factor = 1 + 0.02 * (min(age, 50) - 25)
        # Self-employed variance
        if is_self_employed:
            base_income *= random.uniform(0.5, 1.5)

        wage_income = max(25000, base_income * age_factor)

        # Other income (investment, etc.) - increases with age
        if age > 50:
            other_income = random.uniform(0, 0.2) * wage_income
        else:
            other_income = random.uniform(0, 0.05) * wage_income

        personas.append(
            Persona(
                name=name,
                occupation=occupation,
                filing_status=filing_status,
                wage_income=round(wage_income, 0),
                other_income=round(other_income, 0),
                num_dependents=num_dependents,
                is_self_employed=is_self_employed,
                age=age,
            )
        )

    return personas


def get_factorial_personas(income_levels: List[float]) -> List[Persona]:
    """
    Generate personas for factorial experiment design.

    Creates one persona of each template type at each income level.

    Args:
        income_levels: List of income levels to use

    Returns:
        List of Persona objects
    """
    personas = []

    for template in PERSONA_TEMPLATES:
        for income in income_levels:
            # Adjust other_income based on occupation
            occupation = str(template["occupation"])
            if "Retired" in occupation:
                # Retirees have more investment income
                wage_income = income * 0.4
                other_income = income * 0.6
            else:
                wage_income = income * 0.95
                other_income = income * 0.05

            personas.append(
                Persona(
                    name=str(template["name"]),
                    occupation=occupation,
                    filing_status=template["filing_status"],  # type: ignore[arg-type]
                    wage_income=wage_income,
                    other_income=other_income,
                    num_dependents=int(template["num_dependents"]),
                    is_self_employed=bool(template["is_self_employed"]),
                    age=int(template["age"]),
                )
            )

    return personas
