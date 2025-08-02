#!/usr/bin/env python3
"""Debug simple Gemini response."""

import os

from edsl import Agent, Jobs, Model, QuestionNumerical, Survey


def main():
    """Debug simple Gemini response."""

    if not os.getenv("EXPECTED_PARROT_API_KEY"):
        print("Error: EXPECTED_PARROT_API_KEY not set")
        return

    print("Testing Gemini 2.5 Flash with simple numerical question...")

    # Create a very simple question
    prompt = """You have 20 units of time. Each unit you work gives you $10. 
There is a 25% tax on your earnings.

How many units will you work? (0-20)"""

    question = QuestionNumerical(
        question_name="work_units",
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
    if "answer.work_units" in df.columns:
        answer = df["answer.work_units"].iloc[0]
        print(f"\nWork units answer: {answer}")
        print(f"Type: {type(answer)}")
    else:
        print("\nNo answer found!")

    # Show raw response
    if "raw_model_response.work_units_raw_model_response" in df.columns:
        raw = df["raw_model_response.work_units_raw_model_response"].iloc[0]
        print(f"\nRaw response: {raw}")


if __name__ == "__main__":
    main()
