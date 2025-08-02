#!/usr/bin/env python3
"""Debug Gemini PKNF responses."""

import os

from edsl import Agent, Jobs, Model, QuestionNumerical, Survey


def main():
    """Debug Gemini responses."""

    if not os.getenv("EXPECTED_PARROT_API_KEY"):
        print("Error: EXPECTED_PARROT_API_KEY not set")
        return

    print("Testing Gemini 2.5 Flash with PKNF lab experiment question...")

    # Create a simple lab experiment question
    prompt = """You are participating in an economic experiment (Round 1).

You have a labor endowment of 20 units.
Each unit of labor you supply earns you 20 experimental currency units (ECU).

The tax system is: a flat tax rate of 25%

Note: Under the progressive tax, if you work 20 units you earn 400 ECU and pay 100 ECU in taxes (25%).
If you work 21 units you earn 420 ECU but pay 110 ECU in taxes, leaving you with less after-tax income.

How many units of labor will you supply? (Enter a number between 0 and 20)"""

    question = QuestionNumerical(
        question_name="labor_supply",
        question_text=prompt,
        min_value=0,
        max_value=20,
    )

    survey = Survey([question])

    # Create model
    model = Model("gemini-2.5-flash", service_name="google")

    # Create agent
    agent = Agent(name="Respondent_1")

    # Run survey
    job = Jobs(survey=survey, agents=[agent], models=[model])
    results = job.run()

    # Show raw results
    print("\nRaw results:")
    print(results)

    # Convert to pandas
    df = results.to_pandas()
    print("\nDataFrame columns:")
    print(df.columns.tolist())

    print("\nDataFrame:")
    print(df)

    # Extract answer
    if "answer.labor_supply" in df.columns:
        answer = df["answer.labor_supply"].iloc[0]
        print(f"\nLabor supply answer: {answer}")
        print(f"Type: {type(answer)}")
    else:
        print("\nNo answer.labor_supply column found!")


if __name__ == "__main__":
    main()
