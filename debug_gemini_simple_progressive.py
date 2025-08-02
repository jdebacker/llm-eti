#!/usr/bin/env python3
"""Debug Gemini with simpler progressive tax."""

import os
from edsl import QuestionNumerical, Survey, Jobs, Model, Agent


def main():
    """Debug simpler progressive tax."""

    if not os.getenv("EXPECTED_PARROT_API_KEY"):
        print("Error: EXPECTED_PARROT_API_KEY not set")
        return

    print("Testing Gemini 2.5 Flash with simpler progressive tax question...")

    # Simpler prompt without ECU or experiment framing
    prompt = """You can work up to 20 hours. Each hour pays $20.

Tax rates:
- First $400 of income: 25% tax
- Income above $400: 50% tax

Example: If you work 20 hours, you earn $400 and pay $100 in taxes (25%), keeping $300.
If you work 21 hours, you earn $420 and pay $110 in taxes, keeping $310.

How many hours will you work? (0-20)"""

    question = QuestionNumerical(
        question_name="hours",
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

    # Show results
    df = results.to_pandas()

    if "answer.hours" in df.columns:
        answer = df["answer.hours"].iloc[0]
        print(f"\nHours answer: {answer}")
        print(f"Type: {type(answer)}")
    else:
        print("\nNo answer found!")

    # Show raw response
    if "raw_model_response.hours_raw_model_response" in df.columns:
        raw = df["raw_model_response.hours_raw_model_response"].iloc[0]
        if isinstance(raw, dict) and "candidates" in raw:
            finish_reason = raw["candidates"][0]["finish_reason"]
            text = raw["candidates"][0]["content"]["parts"][0]["text"]
            print(f"\nFinish reason: {finish_reason}")
            print(f"Response text: '{text}'")


if __name__ == "__main__":
    main()
