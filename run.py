from LoadNBAData import NBADataLoader
from NBAPrediction import NBAPredictionModel
from NBAAnalysis import NBAAnalysis
from backtest import run_comparative_backtest
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    """Main function to run the NBA prediction and analysis pipeline."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="NBA Game Predictor")
    parser.add_argument(
        "--load_from_files",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Load data from existing files (default: True)",
    )
    parser.add_argument(
        "--load_new",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Load new data from API (default: True)",
    )
    parser.add_argument(
        "--reload_all",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Reload all data from API (default: False)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train the model (default: 200)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--output_type", type=str, default="csv", help="Output type (default: csv)"
    )
    parser.add_argument(
        "--backtest",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Run backtesting (default: False)",
    )
    parser.add_argument(
        "--only_backtest",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Run only backtesting without predictions (default: False)",
    )
    parser.add_argument(
        "--backtest_interval",
        type=str,
        choices=["day", "week", "month"],
        default="week",
        help="Interval for batch retraining in backtest (default: week)",
    )

    args = parser.parse_args()

    if args.only_backtest:
        print("\nRunning backtests...")
        run_comparative_backtest(args.backtest_interval)
        return

    # Regular prediction pipeline
    data_loader = NBADataLoader(
        load_from_files=args.load_from_files,
        load_new=args.load_new,
        reload_all=args.reload_all,
    )
    data_loader.load_all()

    model = NBAPredictionModel(data_loader)
    model.train(
        data_loader.enhanced_schedule,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    model.prepare_latest_games()
    model.prepare_future_games()

    predictions = model.predict_future_games(model.future_games)

    if args.output_type == "csv":
        predictions.to_csv("nba_predictions.csv", index=False)
    elif args.output_type == "json":
        predictions.to_json("nba_predictions.json", orient="records", date_format="iso")
    print(f"Predictions exported to 'nba_predictions.{args.output_type}'")

    analysis = NBAAnalysis()
    analysis.perform_analysis()
    analysis.print_stats()
    analysis.save_analysis()
    print("Analysis completed and saved to 'nba_display.csv'")

    if args.backtest:
        print("\nRunning backtests...")
        run_comparative_backtest(args.backtest_interval)


if __name__ == "__main__":
    main()
