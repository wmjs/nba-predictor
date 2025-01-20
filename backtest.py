# backtest.py
import pandas as pd
import numpy as np
from NBAPrediction import NBAPredictionModel
from LoadNBAData import NBADataLoader
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta


class NBABacktester:
    def __init__(self, retrain_per_game=False, batch_interval="week"):
        self.data_loader = NBADataLoader(
            load_from_files=True, load_new=False, reload_all=False
        )
        self.model = NBAPredictionModel(self.data_loader, verbose=False)
        self.results = []
        self.retrain_per_game = retrain_per_game
        self.batch_interval = batch_interval

    def evaluate_prediction(self, row, prediction):
        """Evaluate if prediction was correct and calculate relevant metrics"""
        actual_winner = (
            row["Home"] if row["HomePoints"] > row["AwayPoints"] else row["Away"]
        )
        predicted_winner = prediction["PredictedWinner"].iloc[0]
        predicted_spread = prediction["PredictedSpread"].iloc[0]
        actual_spread = abs(row["HomePoints"] - row["AwayPoints"])

        return {
            "Date": row["Date"],
            "Home": row["Home"],
            "Away": row["Away"],
            "ActualWinner": actual_winner,
            "PredictedWinner": predicted_winner,
            "ActualSpread": actual_spread,
            "PredictedSpread": predicted_spread,
            "SpreadError": abs(predicted_spread - actual_spread),
            "Correct": actual_winner == predicted_winner,
            "HomePoints": row["HomePoints"],
            "AwayPoints": row["AwayPoints"],
            "PredictedHomePoints": prediction["PredictedHomePoints"].iloc[0],
            "PredictedAwayPoints": prediction["PredictedAwayPoints"].iloc[0],
        }

    def process_batch(self, batch_data, training_data):
        """Process a batch of games using the same model"""
        # print(f"\nBatch data shape: {batch_data.shape}")
        # print(f"Date range: {batch_data['Date'].min()} to {batch_data['Date'].max()}")

        model = NBAPredictionModel(self.data_loader, verbose=False)
        model.train(training_data)

        batch_results = []
        for _, row in batch_data.iterrows():
            game_to_predict = pd.DataFrame([row])
            prediction = model.predict_future_games(game_to_predict)
            result = self.evaluate_prediction(row, prediction)
            batch_results.append(result)
        return batch_results

    def run_backtest(self):
        schedule_df = pd.read_csv("nba_data/enhanced_schedule.csv")
        schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])
        schedule_df = schedule_df.sort_values("Date")

        valid_games = schedule_df[
            (~pd.isna(schedule_df["HomePoints"]))
            & (~pd.isna(schedule_df["AwayPoints"]))
        ]

        if not self.retrain_per_game:
            # Single model approach
            self.model.train(schedule_df)
            with Pool(processes=cpu_count()) as pool:
                chunk_size = len(valid_games) // cpu_count()
                chunks = [
                    group
                    for _, group in valid_games.groupby(
                        np.arange(len(valid_games)) // chunk_size
                    )
                ]
                results = []
                for chunk in tqdm(chunks, desc="Processing games"):
                    chunk_results = [
                        self.evaluate_prediction(
                            row, self.model.predict_future_games(pd.DataFrame([row]))
                        )
                        for _, row in chunk.iterrows()
                    ]
                    results.extend(chunk_results)
                self.results = results
        else:
            # Batch retraining approach
            if self.batch_interval == "month":
                valid_games["BatchGroup"] = valid_games["Date"].dt.to_period("M")
            elif self.batch_interval == "week":
                valid_games["BatchGroup"] = valid_games["Date"].dt.to_period("W")
            elif self.batch_interval == "day":
                valid_games["BatchGroup"] = valid_games["Date"].dt.to_period("D")

            groups = valid_games.groupby("BatchGroup")

            with Pool(processes=cpu_count()) as pool:
                batch_results = []
                for name, group in tqdm(groups, desc="Processing batches"):
                    training_data = schedule_df[
                        schedule_df["Date"] < group["Date"].min()
                    ]
                    if len(training_data) < 11:
                        continue
                    results = self.process_batch(group, training_data)
                    batch_results.extend(results)

                self.results = batch_results

    def generate_report(self):
        """Generate performance report"""
        results_df = pd.DataFrame(self.results)

        summary = {
            "TotalGames": len(results_df),
            "CorrectPredictions": results_df["Correct"].sum(),
            "Accuracy": results_df["Correct"].mean(),
            "AverageSpreadError": results_df["SpreadError"].mean(),
        }

        results_df.to_csv("backtest_results.csv", index=False)
        pd.DataFrame([summary]).to_csv("backtest_summary.csv", index=False)

        return summary


def run_comparative_backtest(batch_interval="week"):
    """Run both single model and batch retrain backtests"""
    # Run single model backtest
    single_model_backtester = NBABacktester(retrain_per_game=False)
    single_model_backtester.run_backtest()
    single_results = pd.DataFrame(single_model_backtester.results)
    single_summary = single_model_backtester.generate_report()

    # Run batch retrain backtest
    batch_backtester = NBABacktester(
        retrain_per_game=True, batch_interval=batch_interval
    )
    batch_backtester.run_backtest()
    batch_results = pd.DataFrame(batch_backtester.results)
    batch_summary = batch_backtester.generate_report()

    combined_results = pd.merge(
        single_results.add_suffix("_SingleModel"),
        batch_results.add_suffix(f"_{batch_interval}BatchRetrain"),
        left_on=["Date_SingleModel", "Home_SingleModel", "Away_SingleModel"],
        right_on=[
            f"Date_{batch_interval}BatchRetrain",
            f"Home_{batch_interval}BatchRetrain",
            f"Away_{batch_interval}BatchRetrain",
        ],
        suffixes=("_SingleModel", f"_{batch_interval}BatchRetrain"),
        how="left",
    )

    # Create combined summary
    combined_summary = pd.DataFrame(
        {
            "Metric": [
                "TotalGames",
                "CorrectPredictions",
                "Accuracy",
                "AverageSpreadError",
            ],
            "SingleModel": [
                single_summary["TotalGames"],
                single_summary["CorrectPredictions"],
                single_summary["Accuracy"],
                single_summary["AverageSpreadError"],
            ],
            f"{batch_interval}BatchRetrain": [
                batch_summary["TotalGames"],
                batch_summary["CorrectPredictions"],
                batch_summary["Accuracy"],
                batch_summary["AverageSpreadError"],
            ],
        }
    )

    # Save combined results
    combined_results.sort_values(by="Date_SingleModel", inplace=True, ascending=False)
    combined_results.to_csv("backtest_results.csv", index=False)
    combined_summary.to_csv("backtest_summary.csv", index=False)

    # Print results
    print("\nBacktesting Results:")
    print(f"Batch Interval: {batch_interval}")
    print("\nSingle Model:")
    print(f"Accuracy: {single_summary['Accuracy']:.2%}")
    print(f"Average Spread Error: {single_summary['AverageSpreadError']:.2f} points")
    print("\nBatch Retrain:")
    print(f"Accuracy: {batch_summary['Accuracy']:.2%}")
    print(f"Average Spread Error: {batch_summary['AverageSpreadError']:.2f} points")

    return combined_summary


def main():
    parser = argparse.ArgumentParser(description="Run NBA game prediction backtest")
    parser.add_argument(
        "--batch-interval",
        choices=["month", "week", "day"],
        default="month",
        help="Interval for model retraining",
    )
    args = parser.parse_args()

    run_comparative_backtest(args.batch_interval)


if __name__ == "__main__":
    main()
