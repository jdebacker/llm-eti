import re


def calculate_eti(initial_rate, new_rate, initial_income, new_income):
    percent_change_income = (new_income - initial_income) / initial_income
    percent_change_net_of_tax_rate = ((1 - new_rate) - (1 - initial_rate)) / (
        1 - initial_rate
    )
    return percent_change_income / percent_change_net_of_tax_rate


def parse_income_response(response):
    try:
        return float(response.strip().replace("$", "").replace(",", ""))
    except ValueError:
        return None
