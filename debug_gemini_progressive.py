#!/usr/bin/env python3
"""Debug Gemini progressive tax question."""

import os

from edsl import Agent, Jobs, Model, QuestionNumerical, Survey


def main():
    """Debug Gemini progressive tax."""

    if not os.getenv("EXPECTED_PARROT_API_KEY"):
        print("Error: EXPECTED_PARROT_API_KEY not set")
        return

    print("Testing Gemini 2.5 Flash with progressive tax question...")

    # Test the exact prompt that's failing
    prompt = """You are participating in an economic experiment (Round 1).

You have a labor endowment of 20 units.
Each unit of labor you supply earns you 20 experimental currency units (ECU).

The tax system is: a progressive tax where income up to 400 is taxed at 25%, and income above 400 is taxed at 50%

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
    agent = Agent(name="Worker")

    # Run survey
    job = Jobs(survey=survey, agents=[agent], models=[model])
    results = job.run()

    # Show raw results
    df = results.to_pandas()

    # Extract answer
    if "answer.labor_supply" in df.columns:
        answer = df["answer.labor_supply"].iloc[0]
        print(f"\nLabor supply answer: {answer}")
        print(f"Type: {type(answer)}")
    else:
        print("\nNo answer found!")

    # Show raw response
    if "raw_model_response.labor_supply_raw_model_response" in df.columns:
        raw = df["raw_model_response.labor_supply_raw_model_response"].iloc[0]
        print(f"\nRaw response: {raw}")

    # Show validation status
    if "validated.labor_supply_validated" in df.columns:
        validated = df["validated.labor_supply_validated"].iloc[0]
        print(f"\nValidated: {validated}")

    # Show comment
    if "comment.labor_supply_comment" in df.columns:
        comment = df["comment.labor_supply_comment"].iloc[0]
        print(f"\nComment: {comment}")


if __name__ == "__main__":
    main()
