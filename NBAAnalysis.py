import pandas as pd
import numpy as np
import LoadBettingData


class NBAAnalysis:
    def __init__(self):
        self.odds_list = LoadBettingData.load_odds()
        self.pred_df = pd.read_csv("nba_predictions.csv", parse_dates=["Date"])
        self.value_df = self.prepare_value_df()

    def prepare_value_df(self):
        self.value_df = self.pred_df.copy()
        for odds_df in self.odds_list:
            self.value_df = pd.merge(
                self.value_df, odds_df, on=["Date", "Away", "Home"], how="left"
            )

        def book_winner(row):
            if row["Home Money Line Odds"] < row["Away Money Line Odds"]:
                return row["Home"]
            elif row["Home Money Line Odds"] > row["Away Money Line Odds"]:
                return row["Away"]
            else:
                return ""

        self.value_df["Book Winner"] = self.value_df.apply(book_winner, axis=1)
        return self.value_df

    def money_line_value(self, row):
        if row["Book Winner"] == "":
            return "No"
        if row["Book Winner"] == row["PredictedWinner"]:
            return "No"
        else:
            return "Money Line"

    def perform_analysis(self, return_df=False):
        self.value_df["Value?"] = self.value_df.apply(self.money_line_value, axis=1)
        if return_df:
            return self.value_df

    def print_stats(self):
        print(
            f"Number of Money Line Value Bets (Underdog upsets): {len(self.value_df[self.value_df['Value?'] == 'Money Line'])}"
        )

    def save_analysis(self):
        self.value_df.to_csv("nba_display.csv", index=False)
