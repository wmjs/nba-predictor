# NBA Game Predictor

A machine learning model that predicts NBA game outcomes using historical game data and team statistics.

Please note that this project is heavily work in progress. Check out the Limitations section for more details.

## Overview

This project scrapes NBA game data from basketball-reference.com, processes team statistics, and uses a neural network to predict future game outcomes. The model considers various team statistics including offensive/defensive ratings, shooting percentages, and rebounding numbers.

## Features

- Automated data collection from basketball-reference.com
- Historical game data processing and analysis
- Team statistics tracking (both basic and advanced metrics)
- Neural network-based prediction model
- Future game outcome predictions including:
  - Predicted scores
  - Predicted winners
  - Predicted point spreads

## Requirements

- Python 3.8+
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
python NBAPredictor.py
```


This will:
1. Load or scrape NBA data
2. Train the prediction model
3. Generate predictions for upcoming games
4. Export predictions to CSV and JSON files

### Custom Data Loading

The `NBADataLoader` class supports several initialization options:

```python
loader = NBADataLoader(
reload_all=False, # Force reload all data
load_new=False, # Load only new games
load_from_files=False, # Load from saved files instead of web
data_folder='nba_data' # Custom data storage location
)
```


## Data Structure

### Input Features

The model uses the following key statistics:

- Team Advanced Stats:
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

### Output Format

Predictions are exported to both CSV and JSON formats containing:
- Game date
- Home and away teams
- Predicted scores
- Predicted winner
- Predicted point spread

## Model Architecture

The neural network consists of:
- Input layer (based on feature count)
- Three hidden layers (128, 64, and 32 neurons)
- Output layer (2 neurons for home/away scores)
- Batch normalization
- Dropout (0.2)
- ReLU activation

## Data Storage

By default, the following files are created in the data folder:
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

## Notes

- The scraper includes built-in delays to respect basketball-reference.com's rate limits
  - Loading all games from scratch can take a while as the script waits 4 seconds between requests. (4 secs x 300+ games = 20+ minutes).
- Early stopping is implemented to prevent overfitting
- The model automatically handles first games of the season by removing it

## Limitations

- Depends on basketball-reference.com's availability and structure
- Does not account for player injuries or roster changes
- Predictions are based on team-level statistics only
- Updates are not done automatically... Predictions can change drastically week to week.

## Known Issues

- The data loader is not robust and can fail when using `load_new=True`. It is best to use `reload_all=True` when loading new data.
