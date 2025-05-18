# %%
# imports
from openai import OpenAI
import pandas as pd
import numpy as np
import re
import pickle
import time
import os


"""
This Python script accesses the OpenAI API to get responses from
OpenAI's GPT model
"""
# %%
# Get api key from file or prompt user for it
# Check for a file named "gpt_api_key.txt" in the current directory
if os.path.exists(os.path.join("gpt_api_key.txt")):
    with open(os.path.join("gpt_api_key.txt"), "r") as file:
        API_KEY = str(file.read().strip())
else:  # if file not exist, prompt user for token
    try:
        API_KEY = input(
            "Please enter your UN API token (press return if you do not have one): "
        )
        # write the API_KEY to a file to find in the future
        with open(os.path.join("gpt_api_key.txt"), "w") as file:
            file.write(API_KEY)
    except EOFError:
        API_KEY = ""

# %%
# make sure the data directory exists
if not os.path.exists(os.path.join("..", "data")):
    os.makedirs(os.path.join("..", "data"))
# Set number of observations to get from GPT
R = 16  # number of rounds in the tax simulation
R_SWITCH = 8  # round number where switch tax regime


class TaxBehaviorReplication:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the replication study with API key and model choice

        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI model to use for simulation

        Returns:
            None

        """
        self.client = OpenAI(api_key=API_KEY)
        # define file name (a string, path and extension set below)
        self.data_filename = f"PKNF_modified_40pct_results_{model}"
        self.model = model
        self.instructions_text = (
            "You will now participate in a decision-making experiment on "
            + "behavior towards taxation. This experiment has a decision and "
            + "a working stage: \n"
            + "The decision stage consists of 16 rounds. In each round you "
            + "will choose how much income you want to earn. The income "
            + "determines the number of tasks you have to complete later. "
            + "The task is to transcribe words. \n"
            + " You will have to pay taxes on your income. The tax rate "
            + "may, but does not have to, vary from round to round. Each of "
            + "the 16 rounds is independent of each other. \n"
            + "In each round, you are first informed of the tax rate in this "
            + "round and the income that you can earn. The income can be up "
            + "to 600 cents. The higher the income you choose, the more "
            + "tasks you will have to complete. The lower the income, the "
            + "earlier you can finish the experiment. 20 cents correspond to "
            + "1 task. \n"
            + "After you have entered an income, the number of tasks and the "
            + "due tax payment will be automatically calculated and shown on "
            + "the screen. The tax payment equals the chosen income "
            + "multiplied by the tax rate. After each round, you will "
            + "receive information about your payoff. Your payoff is the "
            + "chosen income minus the tax payment. \n"
            + "In the working stage, you will have to complete the number "
            + "of tasks to earn the income that you indicated in one of "
            + "the previous 16 rounds. This round will be randomly selected. "
            + "It also determines how much your additional earnings from the "
            + "experiment will be. \n"
            + "On the following screen, we will explain the working stage in "
            + "more detail. \n"  # New screen below
            + "After the 16 rounds in the decision stage, you will have to "
            + "work on the income you chose in one randomly selected round. \n"
            + "Your task is to transcribe text sequences. Each text "
            + "sequence consists of 10 letters, see the example below. "
            + "The number of tasks that you will work on depends on your "
            + "decisions in the 16 rounds and on chance. A sequence is "
            + "counted when you correctly typed in every letter."
        )
        self.tax_parameterizations = {
            "rate1": [25, 25, 40],
            "rate2": [40, 25, 40],
            "max_labor": [
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
            ],  # will randomly draw from this
        }

    def generate_system_prompt(self, personality_type="neutral"):
        """
        Generate system prompt with different personality types

        Args:
            personality_type (str): Type of personality to generate prompt for

        Returns:
            str: System prompt text

        """
        base_prompt = "You are participating in an economic experiment about taxation and labor supply."

        personality_prompts = {
            "risk_averse": base_prompt
            + " You are very cautious about financial decisions and prefer stable income.",
            "risk_seeking": base_prompt
            + " You are comfortable with financial risks and aim to maximize income.",
            "neutral": base_prompt
            + " You make decisions based on careful evaluation of options.",
        }

        return personality_prompts.get(personality_type, personality_prompts["neutral"])

    def generate_tax_scenario(
        self,
        rate1,
        rate2,
        max_labor,
        bkt1=400,
        wage_rate=20,
        round_number=1,
        max_rounds=R,
    ):
        """
        Generate a tax scenario text for a given round

        Args:
            rate1 (float)): tax rate on first income bracket, in
                percentage points
            rate2 (float): tax rate on all income when above threshold,
                in percentage points
            bkt1 (float): income (in cents) above which rate2 applies
            max_labor (float): maximum units of labor supply
            wage_rate (float): wage rate per unit of labor, in cents
            round_number (int): round number
            max_rounds (int): maximum number of rounds

        Returns:
            sim_text (str): text string to pass to GPT-3 model

        """
        if (
            rate1 == rate2
        ):  # I don't know if PKNF modify instructions this way or not --
            # they only give example of the progressive tax case in appendix
            tax_text = (
                "In this round, the tax rate is "
                + f"{rate1}"
                + "% for all incomes. For example, for an income of "
                + f"{bkt1 + 20} cents, your tax payment will be "
                + f"{(rate1 / 100) * 420:.0f}"
                + " cents."
            )
        else:
            tax_text = (
                "In this round, the tax rate is "
                + f"{rate1}"
                + "% for incomes equal to or below "
                + f"{bkt1}"
                + " cents.  The tax rate is "
                + f"{rate2}"
                + "% on the entire income if income exceeds "
                + f"{bkt1}"
                + f" cents. For example, for an income of {bkt1 + 20} "
                + " cents, your tax payment will be "
                + f"{(rate2 / 100) * (bkt1 + 20):.0f}"
                + " cents."
            )
        sim_text = (
            f"Round {round_number} of {max_rounds} \n"
            + tax_text
            + "\n"
            + f"You can earn an income of {max_labor * wage_rate:.0f}"
            + " cents. \n"
            + "Please indicate whether you want to work for "
            + f"{max_labor * wage_rate:.0f} cents or another income: \n"
            # + "Number of text sequences for this chosen income: "
            # + {chosen_labor} + "\n"  NOTE: This is in original instructions, but not sure how work with LLM
        )

        return sim_text

    def get_tax_rates(self, round_num, treatment):
        """
        Get tax rates for a given round

        Args:
            round_num (int): round number

        Returns:
            tuple: tax rates (rate1, rate2)

        """
        if treatment == 1:  # control group: faces progressive taxes in all rounds
            rate1 = self.tax_parameterizations["rate1"][0]
            rate2 = self.tax_parameterizations["rate2"][0]
        elif treatment == 2:  # treatment 1: moves from progressive to flat tax at 25%
            rate1 = self.tax_parameterizations["rate1"][0]
            rate2 = self.tax_parameterizations["rate2"][0]
            if round_num > R_SWITCH:
                rate1 = self.tax_parameterizations["rate1"][1]
                rate2 = self.tax_parameterizations["rate2"][1]
        elif treatment == 3:  # treatment 2: moves from progressive to flat tax at 40%
            rate1 = self.tax_parameterizations["rate1"][0]
            rate2 = self.tax_parameterizations["rate2"][0]
            if round_num > R_SWITCH:
                rate1 = self.tax_parameterizations["rate1"][2]
                rate2 = self.tax_parameterizations["rate2"][2]
        if (
            treatment == 4
        ):  # treatment that starts with flat tax of 40%, then switches to progressive
            rate1 = self.tax_parameterizations["rate1"][2]
            rate2 = self.tax_parameterizations["rate2"][2]
            if round_num > R_SWITCH:
                rate1 = self.tax_parameterizations["rate1"][0]
                rate2 = self.tax_parameterizations["rate2"][0]
        elif (
            treatment == 5
        ):  # treatment that starts with flat tax of 25%, then switches to progressive
            rate1 = self.tax_parameterizations["rate1"][1]
            rate2 = self.tax_parameterizations["rate2"][1]
            if round_num > R_SWITCH:
                rate1 = self.tax_parameterizations["rate1"][0]
                rate2 = self.tax_parameterizations["rate2"][0]

        return rate1, rate2

    def parse_labor_decision(self, response_list):
        """
        Parse the labor decision from the LLM response

        Args:
            response_list (list): text string responses from LLM

        Returns:
            labor (list): chosen labor supply for each observation in the response

        """
        wage_rate = 20  # cents per task -- hardcoded for now
        income_list = [re.findall(r"\d+", result)[0] for result in response_list]
        labor = [int(income) / wage_rate for income in income_list]

        return labor

    def simulate_labor_decision(
        self,
        rate1,
        rate2,
        max_labor,
        num_obs,
        round_number,
        treatment,
        personality="neutral",
    ):
        """
        Simulate a single labor supply decision

        Args:
            rate1 (float): tax rate on first income bracket, in
                percentage points
            rate2 (float): tax rate on all income when above threshold,
                in percentage points
            max_labor (float): maximum units of labor supply
            num_obs (int): number of observations to get from GPT
            round_number (int): round number
            treatment (int): treatment assignment (1-5)
            personality (str): personality type of the subject

        Returns:
            dict: dictionary containing the results of the simulation

        """
        # find the tax rates based on the round number and treatment
        rate1, rate2 = self.get_tax_rates(round_num=round_number, treatment=treatment)

        # Generate the tax scenario text
        scenario_text = self.generate_tax_scenario(
            rate1, rate2, max_labor, round_number=round_number
        )

        # Get LLM response
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=1.0,
            messages=[
                {"role": "system", "content": self.instructions_text},
                {"role": "user", "content": scenario_text},
            ],
            n=num_obs,
        )

        # Parse the response
        response_list = [choice.message.content for choice in response.choices]
        chosen_labor = self.parse_labor_decision(response_list)

        return {
            "round": [round_number] * num_obs,
            "rate1": [rate1] * num_obs,
            "rate2": [rate2] * num_obs,
            "treatment": [treatment] * num_obs,
            "max_labor": [max_labor * 20] * num_obs,
            "chosen_labor": chosen_labor,
            "personality": [personality] * num_obs,
            "raw_response": response_list,
        }

    def run_full_experiment(
        self,
        num_subjects=100,
        personality_distribution=None,
    ):
        """
        Run the full experiment with multiple subjects

        Args:
            num_subjects (int): number of subjects to simulate per treatment
            personality_distribution (dict): distribution of personality types

        Returns:
            pd.DataFrame: DataFrame containing the results of the experiment

        """
        # if personality_distribution is None:
        #     personality_distribution = {
        #         "neutral": 0.6,
        #         "risk_averse": 0.2,
        #         "risk_seeking": 0.2,
        #     }

        results = {
            "subject_id": [],
            "round": [],
            "rate1": [],
            "rate2": [],
            "treatment": [],
            "max_labor": [],
            "chosen_labor": [],
            "personality": [],
            "raw_response": [],
        }

        # Loop over treatments
        for treatment in range(1, 6):
            # Randomly assign personality type
            # personality = np.random.choice(
            #     list(personality_distribution.keys()),
            #     p=list(personality_distribution.values()),
            # )
            personality = "neutral"

            # Run R rounds for this subject  # TODO: vectorize this
            for round_num in range(1, R + 1):
                # Get tax parameters for this round
                rate1, rate2 = self.get_tax_rates(round_num, treatment)
                # max_labor = np.random.choice(self.tax_parameterizations["max_labor"])
                # When using the "n" argument in the API call, need to pass all the same labor supply, so can't change
                # Instead, we just loop over max_labor so that we get all combinations
                # Since each round is a new chat, there's no history of the previous rounds
                # anyway, so this shouldn't affect the results
                for max_labor in self.tax_parameterizations["max_labor"]:
                    result = self.simulate_labor_decision(
                        rate1=rate1,
                        rate2=rate2,
                        max_labor=max_labor,
                        num_obs=num_subjects,
                        round_number=round_num,
                        treatment=treatment,
                        personality=personality,
                    )

                    result["subject_id"] = np.arange(1, num_subjects + 1).tolist()
                    for k in results.keys():
                        results[k].extend(result[k])

                # Rate limiting
                time.sleep(0.1)

        return pd.DataFrame(results)


def main():
    # Initialize experiment
    experiment = TaxBehaviorReplication(model="gpt-4.1-mini-2025-04-14")

    # Run experiment
    results_df = experiment.run_full_experiment(
        num_subjects=128,
        # personality_distribution={
        #     "neutral": 0.5,
        #     "risk_averse": 0.25,
        #     "risk_seeking": 0.25,
        # },
    )

    # Save results to disk as CSV and DataFrame via pickle
    results_df.to_csv(os.path.join("..", "data", experiment.data_filename + ".csv"))
    pickle.dump(
        results_df,
        open(os.path.join("..", "data", experiment.data_filename + ".pkl"), "wb"),
    )


if __name__ == "__main__":
    main()

