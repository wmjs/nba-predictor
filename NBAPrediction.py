from LoadNBAData import NBADataLoader
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

import argparse

class NBADataset(Dataset):
    """PyTorch Dataset for NBA game data.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NBAPredictor(nn.Module):
    """Neural network model for predicting NBA game scores.
    
    Args:
        input_size (int): Number of input features
    """
    def __init__(self, input_size):
        super(NBAPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.layer4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout(self.relu(self.bn2(self.layer2(x))))
        x = self.dropout(self.relu(self.bn3(self.layer3(x))))
        return self.layer4(x)
    
class NBAPredictionModel:
    """Main class for NBA game prediction pipeline.
    
    Args:
        data (NBADataLoader, optional): Data loader object containing NBA game data
    """
    def __init__(self, data = None):
        self.feature_columns = [
            # Team advanced stats
            'Away_advanced_ORtg', 'Away_advanced_DRtg',
            'Home_advanced_ORtg', 'Home_advanced_DRtg',

            # Shooting stats
            'Away_basic_FG%', 'Away_basic_3P%', 'Away_basic_FT%',
            'Home_basic_FG%', 'Home_basic_3P%', 'Home_basic_FT%',

            # Shooting Rates
            'Away_basic_3PA', 'Away_basic_FTA',
            'Home_basic_3PA', 'Home_basic_FTA',

            # Rebounding
            'Away_basic_TRB', 'Away_basic_ORB', 'Away_basic_DRB',
            'Home_basic_TRB', 'Home_basic_ORB', 'Home_basic_DRB',
        ]
        self.target_columns = ['AwayPoints', 'HomePoints']
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.enhanced_schedule = data.enhanced_schedule
        self.latest_game_stats = None
        self.schedule_no_results = data.schedule_no_results
        self.data_folder = data.data_folder

    def train(self, df, num_epochs=200, batch_size=16):
        """Train the neural network model.
        
        Args:
            df (pandas.DataFrame): Training data
            num_epochs (int, optional): Number of training epochs. Defaults to 200
            batch_size (int, optional): Batch size for training. Defaults to 16
            
        Returns:
            torch.nn.Module: Trained model in evaluation mode
        """
        X = df[self.feature_columns].values
        y = df[self.target_columns].values
        
        # Scale data
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)

        # Create data loaders
        train_dataset = NBADataset(torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled))
        test_dataset = NBADataset(torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize model and training components
        self.model = NBAPredictor(len(self.feature_columns))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            avg_val_loss = val_loss / len(test_loader)
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.data_folder+'/best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

                # Load best model
        self.model.load_state_dict(torch.load(self.data_folder+'/best_model.pth', weights_only=True))
        return self.model.eval()
            
    def predict_future_games(self, future_games):
        """Make predictions for upcoming games.
        
        Args:
            future_games (pandas.DataFrame): DataFrame containing future game data
            
        Returns:
            pandas.DataFrame: Formatted predictions including scores and winners
        """
        future_X = future_games[self.feature_columns].values
        future_X_scaled = self.feature_scaler.transform(future_X)
        future_X_tensor = torch.FloatTensor(future_X_scaled)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(future_X_tensor)
            predictions_original = self.target_scaler.inverse_transform(predictions.numpy())

        return self._format_predictions(future_games, predictions_original)

    def _format_predictions(self, future_games, predictions):
        future_games = future_games.copy()
        future_games['PredictedAwayPoints'] = predictions[:, 0]
        future_games['PredictedHomePoints'] = predictions[:, 1]
        future_games['PredictedSpread'] = future_games.apply(self._predicted_spread, axis=1)
        future_games['PredictedWinner'] = future_games.apply(self._predicted_winner, axis=1)

        prediction_display = future_games[['Date', 'Away', 'PredictedAwayPoints', 
                                         'Home', 'PredictedHomePoints', 'PredictedWinner', 'PredictedSpread']].copy()
        
        prediction_display['PredictedAwayPoints'] = prediction_display['PredictedAwayPoints'].round(1)
        prediction_display['PredictedHomePoints'] = prediction_display['PredictedHomePoints'].round(1)
        prediction_display['PredictedSpread'] = prediction_display['PredictedSpread'].round(1)
        prediction_display['Date'] = pd.to_datetime(prediction_display['Date'])

        # Calculate confidence levels based on PredictedSpread
        max_spread = prediction_display['PredictedSpread'].max()
        min_spread = prediction_display['PredictedSpread'].min()
        
        # Linear scaling between 50% and 95% confidence
        prediction_display['Confidence'] = prediction_display['PredictedSpread'].apply(
            lambda x: round(50 + (x - min_spread) * (95 - 50) / (max_spread - min_spread), 2)
        )
        
        return prediction_display.sort_values('Date')
    
    def prepare_latest_games(self):
        """Prepare the most recent game statistics for each team."""
        latest_home_games = self.enhanced_schedule.sort_values('Date').groupby('Home').last().reset_index()
        latest_away_games = self.enhanced_schedule.sort_values('Date').groupby('Away').last().reset_index()

        # Combine home and away games and get the latest for each team
        latest_games = pd.concat([
            latest_home_games[['Home', 'Date']].rename(columns={'Home': 'Team'}).assign(Location='Home'),
            latest_away_games[['Away', 'Date']].rename(columns={'Away': 'Team'}).assign(Location='Away')
        ])
        latest_games = latest_games.sort_values('Date').groupby('Team').last().reset_index()

        latest_game_stats = pd.DataFrame()
        for tup in latest_games.itertuples():
            prefix = tup.Location+"_"
            cols = [col for col in self.enhanced_schedule.columns if str(col).startswith(prefix)]
            latest_game_stats_team = self.enhanced_schedule[(self.enhanced_schedule[tup.Location] == tup.Team) & (self.enhanced_schedule["Date"] == tup.Date)][cols]
            latest_game_stats_team["Team"] = tup.Team
            latest_game_stats_team.rename(columns={col: col.replace(prefix, "") for col in latest_game_stats_team.columns}, inplace=True)
            latest_game_stats = pd.concat([latest_game_stats, latest_game_stats_team])
        
        self.latest_game_stats = latest_game_stats

    def prepare_future_games(self):
        """Prepare upcoming games data by merging with latest team statistics."""
        future_games = self.schedule_no_results.merge(
            self.latest_game_stats.add_prefix('Away_'), 
            left_on="Away", 
            right_on="Away_Team", 
            how="left"
        ).drop(columns=["Away_Team"])\
        .merge(
            self.latest_game_stats.add_prefix('Home_'), 
            left_on="Home", 
            right_on="Home_Team", 
            how="left"
        ).drop(columns=["Home_Team"])
        self.future_games = future_games
    
    @staticmethod
    def _predicted_winner(row):
        return row['Home'] if row['PredictedHomePoints'] > row['PredictedAwayPoints'] else row['Away']

    @staticmethod
    def _predicted_spread(row):
        return abs(row['PredictedHomePoints'] - row['PredictedAwayPoints'])