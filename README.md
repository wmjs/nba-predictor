# NBA Game Predictor

A machine learning model that predicts NBA game outcomes using historical game data and team statistics from basketball-reference.com.

## Overview

This project uses a neural network to predict NBA game outcomes by analyzing team performance metrics. It automatically scrapes and processes game data, maintains historical statistics, and generates predictions for upcoming games.

## Features

- Automated data collection from basketball-reference.com
- Historical game data processing and analysis
- Team statistics tracking (both basic and advanced metrics)
- Neural network-based prediction model
- Configurable data loading options (new games only, full reload, or from saved files)
- Early stopping to prevent overfitting
- Batch normalization and dropout for improved model stability
- Predictions include:
  - Expected scores
  - Predicted winners
  - Point spreads

## Requirements

- Python
- PyTorch
- pandas
- numpy
- scikit-learn
- tqdm
- html5lib
- beautifulsoup4

Install dependencies using:

```bash
pip install -r requirements.txt
```


## Usage

### Basic Usage

Run the predictor with default settings:

```bash
python NBAPrediction.py
```


### Advanced Usage

The script supports several command-line arguments:

```bash
python NBAPrediction.py [--load_from_files BOOL] [--load_new BOOL] [--reload_all BOOL] [--num_epochs INT] [--batch_size INT]
```


Arguments:
- `--load_from_files`: Load data from saved files (default: True)
- `--load_new`: Load only new games (default: True)
- `--reload_all`: Force reload all data (default: False)
- `--num_epochs`: Number of training epochs (default: 200)
- `--batch_size`: Training batch size (default: 16)

## Data Structure

### Input Features

The model uses the following team statistics:

- Advanced Stats:
  - Offensive Rating (ORtg)
  - Defensive Rating (DRtg)
- Shooting Stats:
  - Field Goal Percentage (FG%)
  - Three-Point Percentage (3P%)
  - Free Throw Percentage (FT%)
- Shot Selection:
  - Three-Point Attempts (3PA)
  - Free Throw Attempts (FTA)
- Rebounding:
  - Total Rebounds (TRB)
  - Offensive Rebounds (ORB)
  - Defensive Rebounds (DRB)

### Model Architecture

The neural network consists of:
- Input layer (based on feature count)
- Three hidden layers (128, 64, and 32 neurons)
- Output layer (2 neurons for home/away scores)
- Batch normalization after each hidden layer
- Dropout (0.2) for regularization
- ReLU activation functions

## Data Storage

The project creates the following files in the data folder:

- `nba_team_abbreviations.csv`: Team name mappings
- `schedule_and_results.csv`: Game schedule and results
- `combined_schedule_and_game_stats.csv`: Detailed game statistics
- `home_games_stats.csv`: Home game statistics
- `away_games_stats.csv`: Away game statistics
- `all_team_stats.csv`: Combined team statistics
- `enhanced_schedule.csv`: Schedule with added team statistics
- `best_model.pth`: Saved model weights
- `nba_predictions.csv`: Latest predictions
- `nba_predictions.json`: Latest predictions (JSON format)

## Notes and Limitations

### Data Collection
- Includes built-in delays (4 seconds between requests) to respect basketball-reference.com's rate limits
- Loading all games from scratch can take 20+ minutes due to rate limiting
- First games of the season are automatically excluded from training data

### Model Limitations
- Does not account for:
  - Player injuries
  - Roster changes
  - Rest days
  - Travel schedule
  - Team momentum
- Predictions are based solely on team-level statistics
- Model needs regular updates as team performance changes throughout the season

### Known Issues
- The data loader can be unstable when using `load_new=True`. (I think I resolved this but not sure...)
- Recommended to use `reload_all=True` when loading new data
- Some game statistics may be missing if basketball-reference.com's structure changes

## Contributing

Feel free to open issues or submit pull requests for improvements. Areas that need work:
- Robust error handling for data scraping
- Additional features for player-level statistics
- Improved prediction accuracy through feature engineering
- Automated daily updates
