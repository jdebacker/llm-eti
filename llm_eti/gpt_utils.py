from typing import List

from openai import OpenAI


class GPTClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_gpt4_response(self, prompt: str, n: int = 1) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            print(f"Error getting GPT response: {str(e)}")
            return []

    @staticmethod
    def calculate_eti(
        initial_rate: float,
        new_rate: float,
        initial_income: float,
        new_income: float,
    ) -> float:
        try:
            percent_change_income = (new_income - initial_income) / initial_income
            percent_change_net_of_tax_rate = ((1 - new_rate) - (1 - initial_rate)) / (
                1 - initial_rate
            )
            if percent_change_net_of_tax_rate == 0:
                return None
            return percent_change_income / percent_change_net_of_tax_rate
        except ZeroDivisionError:
            return None
        except Exception as e:
            print(f"Error calculating ETI: {str(e)}")
            return None

    @staticmethod
    def parse_income_response(response: str) -> float:
        """Parse the income response, handling various formats."""
        try:
            # Remove common currency symbols and formatting
            cleaned = response.strip().replace("$", "").replace(",", "")
            # Convert to float and handle negative numbers
            return abs(float(cleaned))
        except (ValueError, TypeError):
            return None
