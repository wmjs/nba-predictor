# NBA Game Predictor

An automated NBA game prediction system that uses machine learning to forecast game outcomes and identify potential betting opportunities.

## Overview

This project combines historical NBA game data, team statistics, and betting odds to:
- Predict outcomes of upcoming NBA games
- Calculate win probabilities and expected point spreads
- Identify value betting opportunities by comparing predictions to market odds
- Automatically update predictions daily during the NBA season

## Features

- **Data Collection**
  - Scrapes game statistics from Basketball Reference
  - Retrieves current betting odds from Sportsline
  - Maintains historical team performance metrics
  - Updates data automatically via GitHub Actions

- **Prediction Model**
  - Uses neural network architecture for game outcome prediction
  - Considers both basic and advanced team statistics
  - Accounts for home/away performance differentials
  - Generates point spread and total predictions

- **Analysis Tools**
  - Compares predicted outcomes to betting market odds
  - Calculates expected value for betting opportunities
  - Tracks prediction accuracy over time
  - Exports results to CSV for easy analysis

## Data Sources

- Game statistics: basketball-reference.com
- Betting odds: sportsline.com
- Team abbreviations and metadata maintained in project

## Technical Details

### Requirements
- Python 3.x
- pandas
- numpy
- tqdm
- PyTorch (for neural network model)
- requests/beautifulsoup4 (for web scraping)

### Key Components
- `LoadNBAData.py`: Data collection and processing
- `NBAPrediction.py`: Machine learning model implementation
- `LoadBettingData.py`: Odds data retrieval and processing
- `NBAAnalysis.py`: Results analysis and reporting

### Automation
- Daily data updates via GitHub Actions
- Automated model retraining with new game data
- Scheduled prediction updates for upcoming games

## Usage

1. Clone the repository
2. Install required dependencies
3. Run prediction pipeline:

```bash
python run.py
```

## Data Files

- `all_team_stats.csv`: Comprehensive team statistics
- `away_games_stats.csv`: Away game performance metrics
- `home_games_stats.csv`: Home game performance metrics
- `schedule_and_results.csv`: Game schedule and outcomes
- `enhanced_schedule.csv`: Schedule with additional metrics
- `nba_team_abbreviations.csv`: Team metadata

The script supports several command-line arguments:

```bash
python run.py [--load_from_files BOOL] [--load_new BOOL] [--reload_all BOOL] [--num_epochs INT] [--batch_size INT] [--output_type STR]
```


Arguments:
- `--load_from_files`: Load data from saved files (default: True)
- `--load_new`: Load only new games (default: True)
- `--reload_all`: Force reload all data (default: False)
- `--num_epochs`: Number of training epochs (default: 200)
- `--batch_size`: Training batch size (default: 16)
- `--output_type`: Format for prediction output - 'csv' or 'json' (default: 'csv')

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
- The data loader can be unstable when using `load_new=True`. 
- Recommended to use `reload_all=True` when loading new data
- Some game statistics may be missing if basketball-reference.com's structure changes

This project is for educational purposes only. Please check local regulations regarding sports betting.
