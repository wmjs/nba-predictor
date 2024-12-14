import pandas as pd
import time
from tqdm import tqdm
import os

class NBADataLoader:
    """A class to load and process NBA game data and statistics.

    This class handles fetching, processing, and storing NBA game data including schedules,
    results, and team statistics from basketball-reference.com. It can load data from web
    sources or local files, and supports incremental updates for new games.

    Attributes:
        nba_abbreviations (DataFrame): Team name to abbreviation mappings
        schedule_and_results_raw (DataFrame): Raw schedule and results data
        schedule_and_results (DataFrame): Processed schedule and results
        schedule_no_results (DataFrame): Upcoming games without results
        combined_stats (DataFrame): Combined game statistics
        home_games_df (DataFrame): Statistics for home games
        away_games_df (DataFrame): Statistics for away games
        all_team_stats (DataFrame): Cumulative team statistics
        enhanced_schedule (DataFrame): Schedule with added team statistics
        reload_all (bool): Whether to reload all data from web
        load_new (bool): Whether to load only new games
        latest_combined_stats_date (datetime): Latest date in combined stats
        load_from_files (bool): Whether to load from saved files
        data_folder (str): Directory for data storage

    Args:
        reload_all (bool, optional): Force reload all data. Defaults to False.
        load_new (bool, optional): Load only new games. Defaults to False.
        load_from_files (bool, optional): Load from saved files. Defaults to False.
        data_folder (str, optional): Data storage directory. Defaults to 'nba_data'.
    """

    def __init__(self, reload_all = False, load_new = False, load_from_files = False, data_folder='nba_data'):
        self.nba_abbreviations = None
        self.schedule_and_results_raw = None    
        self.schedule_and_results = None
        self.schedule_no_results = None
        self.combined_stats = None
        self.home_games_df = None
        self.away_games_df = None
        self.all_team_stats = None
        self.enhanced_schedule = None
        self.reload_all = reload_all
        self.load_new = load_new
        self.latest_combined_stats_date = None
        self.load_from_files = load_from_files
        self.data_folder = data_folder + "/"
        
        # Create the data folder if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def get_nba_team_abbreviations(self):
        """Fetch and process NBA team abbreviations from Wikipedia.
        
        Returns:
            DataFrame: Team abbreviations indexed by team name.
        """
        self.nba_abbreviations = (pd.read_html(
            "https://en.wikipedia.org/wiki/Wikipedia:WikiProject_National_Basketball_Association/National_Basketball_Association_team_abbreviations")[0]
            .rename(columns={
                "Franchise": "Team",
                "Abbreviation/ Acronym": "Abbreviation"
            })
            .set_index("Team"))
        self.nba_abbreviations["Abbreviation"] = self.nba_abbreviations["Abbreviation"].str.split().str[0]
        self.nba_abbreviations.loc["Phoenix Suns"] = "PHO"
        self.nba_abbreviations.loc["Charlotte Hornets"] = "CHO"
        self.nba_abbreviations.loc["Brooklyn Nets"] = "BRK"
        return self.nba_abbreviations

    def get_raw_schedule_and_results(self):
        """Fetch NBA schedule and completed game results from basketball-reference.com.
        
        Fetches data month by month for the current season, processes it to remove
        unnecessary columns, and combines into a single DataFrame.

        Returns:
            DataFrame: Raw schedule and results data.
        """
        season_months = ["october", "november", "december", "january", "february", "march", "april"]
        schedule_and_results_raw = pd.DataFrame()
        for month in season_months:    
            # Get schedule data
            schedule_and_results_raw_month = (pd.read_html(
                f"https://www.basketball-reference.com/leagues/NBA_2025_games-{month}.html",
                flavor='html5lib', 
                header=0)[0]
                .rename(columns={
                "Visitor/Neutral": "Away",
                "Home/Neutral": "Home",
                "PTS": "AwayPoints",
                "PTS.1": "HomePoints"
            }))
            if schedule_and_results_raw_month["HomePoints"].isnull().all():
                print(f"No data found for {month}")
                continue
            schedule_and_results_raw_month = schedule_and_results_raw_month[schedule_and_results_raw_month["HomePoints"].notna()]
            schedule_and_results_raw = pd.concat([schedule_and_results_raw, schedule_and_results_raw_month])
        # Clean up columns
        columns_to_drop = ["Unnamed: 6", "Unnamed: 7", "Notes", "Attend.", "LOG", "Arena", "Start (ET)"]
        schedule_and_results_raw.drop(columns=columns_to_drop, inplace=True)
        self.schedule_and_results_raw = schedule_and_results_raw
        print("DATE1", pd.to_datetime(schedule_and_results_raw["Date"]).max(), pd.to_datetime(schedule_and_results_raw["Date"]).min())
        return self.schedule_and_results_raw
    
    def get_schedule_no_results(self):
        """Fetch upcoming NBA games without results.
        
        Similar to get_raw_schedule_and_results but only returns future games
        that haven't been played yet.

        Returns:
            DataFrame: Upcoming games schedule.
        """
        season_months = ["october", "november", "december", "january", "february", "march", "april"]
        schedule_no_results_raw = pd.DataFrame()
        for month in season_months:    
            schedule_no_results_raw_month = (pd.read_html(
                f"https://www.basketball-reference.com/leagues/NBA_2025_games-{month}.html",
                flavor='html5lib', 
                header=0)[0]
                .rename(columns={
                "Visitor/Neutral": "Away",
                "Home/Neutral": "Home",
                "PTS": "AwayPoints",
                "PTS.1": "HomePoints"
            }))
            schedule_no_results_raw_month = schedule_no_results_raw_month[schedule_no_results_raw_month["HomePoints"].isnull()]
            schedule_no_results_raw = pd.concat([schedule_no_results_raw, schedule_no_results_raw_month])
        # Clean up columns
        columns_to_drop = ["Unnamed: 6", "Unnamed: 7", "Notes", "Attend.", "LOG", "Arena", "Start (ET)"]
        schedule_no_results_raw.drop(columns=columns_to_drop, inplace=True)
        self.schedule_no_results = schedule_no_results_raw
        return self.schedule_no_results

    def set_up_schedule_and_results(self):
        """Processes NBA schedule and results data."""
        if self.schedule_and_results_raw is None or self.nba_abbreviations is None:
            raise ValueError("Must call get_raw_schedule_and_results() and get_nba_team_abbreviations() first")
        
        def get_first_game_dates(df):
            """Get the first game date for each team, both home and away."""
            first_home = df.groupby('Home')['Date'].min().reset_index().set_index('Home')
            first_away = df.groupby('Away')['Date'].min().reset_index().set_index('Away')

            first_games = pd.merge(first_home, first_away, left_index=True, right_index=True)
            first_games.columns = ["FirstGameHome", "FirstGameAway"]
            first_games['FirstGame'] = first_games[["FirstGameHome", "FirstGameAway"]].min(axis=1)
            
            return first_games[['FirstGame']]

        def is_first_game(row, first_games):
            """Check if a game is the first game for either team."""
            return (row["Date"] == first_games.loc[row['Home']].values[0] or 
                    row["Date"] == first_games.loc[row['Away']].values[0])
        
        def get_game_winner(row):
            """Get the winner of a game."""
            return "Home" if row["HomePoints"] > row["AwayPoints"] else "Away"
        
        def get_point_differential(row):
            """Get the point differential of a game."""
            return row["HomePoints"] - row["AwayPoints"]
        
        # Process dates and add team abbreviations
        self.schedule_and_results_raw["Date"] = pd.to_datetime(self.schedule_and_results_raw["Date"]).dt.strftime("%Y%m%d")
        self.schedule_and_results_raw["HomeAbbreviation"] = self.schedule_and_results_raw["Home"].map(self.nba_abbreviations["Abbreviation"])
        self.schedule_and_results_raw["AwayAbbreviation"] = self.schedule_and_results_raw["Away"].map(self.nba_abbreviations["Abbreviation"])

        # Generate game URLs
        base_game_url = "https://www.basketball-reference.com/boxscores/{date}0{home_team}.html"
        self.schedule_and_results_raw["GameUrl"] = self.schedule_and_results_raw.apply(
            lambda x: base_game_url.format(
                date=x["Date"],
                home_team=x["HomeAbbreviation"]
            ), axis=1)

        # Remove first games of the season
        first_games = get_first_game_dates(self.schedule_and_results_raw)
        self.schedule_and_results_raw["FirstGame"] = self.schedule_and_results_raw.apply(lambda row: is_first_game(row, first_games), axis=1)
        self.schedule_and_results_raw = self.schedule_and_results_raw[~self.schedule_and_results_raw["FirstGame"]].reset_index(drop=True)

        # Add consistent team ordering columns
        self.schedule_and_results_raw["A_Team"] = self.schedule_and_results_raw[["Home", "Away"]].min(axis=1)
        self.schedule_and_results_raw["B_Team"] = self.schedule_and_results_raw[["Home", "Away"]].max(axis=1)

        # Drop first game column
        self.schedule_and_results_raw.drop(columns=["FirstGame"], inplace=True)

        # Add winner and point differential columns
        self.schedule_and_results_raw["Winner"] = self.schedule_and_results_raw.apply(get_game_winner, axis=1)
        self.schedule_and_results_raw["PointDifferential"] = self.schedule_and_results_raw.apply(get_point_differential, axis=1)

        self.schedule_and_results = self.schedule_and_results_raw
        self.schedule_and_results["Date"] = pd.to_datetime(self.schedule_and_results["Date"])
        return self.schedule_and_results

    def scrape_game_page(self, url):
        """Scrapes basic and advanced stats for both teams."""
        # Read all tables from the page
        table_list = pd.read_html(url)
        
        # Helper function to process team stats table
        def process_team_stats(table, prefix):
            df = (table.droplevel(0, axis=1)
                   .query("Starters == 'Team Totals'")
                   .dropna(axis=1)
                   .drop(columns=['Starters'])
                   .add_prefix(prefix)
                   .reset_index(drop=True))
            
            # Convert all columns to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        
        # Create dictionary with processed stats for each team
        away_basic_stats_index = 0
        home_basic_stats_index = int(len(table_list)/2)
        away_advanced_stats_index = int(len(table_list)/2 - 1)
        home_advanced_stats_index = int(len(table_list) - 1)
        stat_dict = {
            'away_basic_stats': process_team_stats(table_list[away_basic_stats_index], 'away_basic_'),
            'home_basic_stats': process_team_stats(table_list[home_basic_stats_index], 'home_basic_'),
            'away_advanced_stats': process_team_stats(table_list[away_advanced_stats_index], 'away_advanced_'),
            'home_advanced_stats': process_team_stats(table_list[home_advanced_stats_index], 'home_advanced_')
        }

        return stat_dict

    def combine_schedule_and_game_stats(self, schedule):
        """Combines schedule and game stats data."""
        if self.schedule_and_results is None:
            raise ValueError("Must call set_up_schedule_and_results() first")
        
        # Initialize empty lists to store stats for each game
        all_game_stats = []
        num_games = schedule.shape[0]

        # Create progress bar
        pbar = tqdm(total=num_games, desc="Loading game stats")
        
        # Iterate through each game in schedule_and_results
        for i, game in schedule.iterrows():
            try:
                # Add delay between requests to avoid rate limiting
                time.sleep(4)
                
                # Scrape stats from the game URL
                game_stats = self.scrape_game_page(game['GameUrl'])
                
                # Combine all stats DataFrames for this game
                combined_stats = pd.concat([
                    game_stats['away_basic_stats'],
                    game_stats['away_advanced_stats'],
                    game_stats['home_basic_stats'], 
                    game_stats['home_advanced_stats']
                ], axis=1)
                
                # Add all columns from the original schedule row
                for col in schedule.columns:
                    combined_stats[col] = game[col]
                
                all_game_stats.append(combined_stats)

                if len(combined_stats.columns) > 80:
                    print(combined_stats.columns.shape)
                    print(game["GameUrl"])
                
            except Exception as e:
                print(f"\nError processing game {game['Away']} vs {game['Home']}: {str(e)}")
                print(game["GameUrl"])
                # On error, wait longer before next request
                time.sleep(10)
                continue
            finally:
                pbar.update(1)
        
        pbar.close()

        # Combine all games into one DataFrame
        all_games_df = pd.concat(all_game_stats, ignore_index=True)
        # self.combined_stats = all_games_df
        return all_games_df

    def load_combined_schedule_and_game_stats(self, reload=False):
        if reload or self.load_new:
            if self.load_new and not reload:
                # Load existing data
                try:
                    # self.load_all_from_files()
                    self.latest_combined_stats_date = self.get_latest_combined_stats_date()
                    
                    # Get new games only
                    print(f"Last loaded: {self.latest_combined_stats_date}, loading as of today: {self.schedule_and_results['Date'].max()}")
                    new_schedule = self.schedule_and_results[
                        pd.to_datetime(self.schedule_and_results["Date"], format="%Y%m%d") > self.latest_combined_stats_date
                    ]

                    if len(new_schedule) == 0:
                        print("No new games to load")
                        return self.combined_stats
                    
                    print(f"Loading {len(new_schedule)} new games")
                    new_stats = self.combine_schedule_and_game_stats(new_schedule)
                    self.schedule_and_results = pd.concat([new_schedule, self.schedule_and_results], ignore_index=True)
                    self.schedule_and_results.drop_duplicates(keep='first', inplace=True)
                    
                    # Combine old and new data
                    self.combined_stats = pd.concat([self.combined_stats, new_stats], ignore_index=True).drop_duplicates(keep='first')
                    self.combined_stats.drop_duplicates(keep='first', inplace=True)
                    self.save_combined_schedule_and_game_stats()
                    return self.combined_stats
                    
                except FileNotFoundError:
                    print("No existing data found, loading all games")
                    self.combined_stats = self.combine_schedule_and_game_stats(self.schedule_and_results)
                    self.save_combined_schedule_and_game_stats()
                    return self.combined_stats
            else:
                self.combined_stats = self.combine_schedule_and_game_stats(self.schedule_and_results)
                self.save_combined_schedule_and_game_stats()
                return self.combined_stats
        else:
            try:
                self.combined_stats = pd.read_csv(os.path.join(self.data_folder, "combined_schedule_and_game_stats.csv"), index_col=0)
                self.combined_stats["Date"] = pd.to_datetime(self.combined_stats["Date"], format="%Y%m%d")
            except FileNotFoundError:
                raise ValueError("Combined stats file not found, reload")
            return self.combined_stats
        
    def get_latest_combined_stats_date(self):
        if self.combined_stats is None:
            raise ValueError("Must load combined stats first")
        return self.combined_stats["Date"].max()

    def save_combined_schedule_and_game_stats(self):
        filepath = os.path.join(self.data_folder, "combined_schedule_and_game_stats.csv")
        self.combined_stats.to_csv(filepath)

    def split_home_away_games(self):
        """Split combined game statistics into separate home and away dataframes."""
        if self.combined_stats is None:
            raise ValueError("Must load combined stats first")
        
        # Create copies to avoid modifying original
        home_games = self.combined_stats.copy().drop_duplicates(keep='first')
        away_games = self.combined_stats.copy().drop_duplicates(keep='first')

        # Filter and process home games
        home_columns = [col for col in home_games.columns if col.startswith('home_')] + ['Home', 'Date']
        home_games = home_games[home_columns]
        home_games = home_games.rename(columns={'Home': 'Team'})
        home_games = home_games.set_index(['Date', 'Team'])

        # Filter and process away games
        away_columns = [col for col in away_games.columns if col.startswith('away_')] + ['Away', 'Date'] 
        away_games = away_games[away_columns]
        away_games = away_games.rename(columns={'Away': 'Team'})
        away_games = away_games.set_index(['Date', 'Team'])

        self.home_games_df = home_games
        self.away_games_df = away_games
        return self.home_games_df, self.away_games_df

    def calculate_cumulative_team_stats(self):
        """Calculate cumulative statistics for both home and away games."""
        if self.home_games_df is None or self.away_games_df is None:
            raise ValueError("Must call split_home_away_games() first")
        
        # Create copies of home_games_df and away_games_df to avoid modifying originals
        cumulative_home_stats = self.home_games_df.copy()
        cumulative_away_stats = self.away_games_df.copy()

        # Sort by Date within each Team group
        cumulative_home_stats = cumulative_home_stats.groupby('Team', group_keys=False).apply(lambda x: x.sort_index(level='Date'))
        cumulative_away_stats = cumulative_away_stats.groupby('Team', group_keys=False).apply(lambda x: x.sort_index(level='Date'))

        # Calculate cumulative averages for each numeric column within each team group
        numeric_columns_home = cumulative_home_stats.select_dtypes(include=['int64', 'float64']).columns
        numeric_columns_away = cumulative_away_stats.select_dtypes(include=['int64', 'float64']).columns

        cumulative_home_stats[numeric_columns_home] = cumulative_home_stats.groupby('Team')[numeric_columns_home].expanding().mean().reset_index(0, drop=True)
        cumulative_away_stats[numeric_columns_away] = cumulative_away_stats.groupby('Team')[numeric_columns_away].expanding().mean().reset_index(0, drop=True)

        # Function to remove prefixes from column names
        def remove_prefix(df, prefix):
            return df.rename(columns={col: col.replace(prefix, '') for col in df.columns if col.startswith(prefix)})

        # Remove prefixes from both dataframes
        clean_home_stats = remove_prefix(cumulative_home_stats, 'home_')
        clean_away_stats = remove_prefix(cumulative_away_stats, 'away_')

        # Combine home and away stats
        all_team_stats = pd.concat([clean_home_stats, clean_away_stats])

        # Sort by Team and Date and remove redundant index level
        all_team_stats = all_team_stats.sort_index()
        # all_team_stats = all_team_stats.reset_index()
        # all_team_stats = all_team_stats.set_index(['Team', 'Date'])
        
        self.all_team_stats = all_team_stats
        return self.all_team_stats

    def enhance_schedule_with_team_stats(self):
        """Enhance schedule data by adding latest team statistics."""
        if self.schedule_and_results is None or self.all_team_stats is None:
            raise ValueError("Must have schedule and team stats calculated first")
        
        # Create a copy of schedule_and_results to avoid modifying the original
        enhanced_schedule = self.schedule_and_results.copy()

        # Function to get the latest stats before a given date
        def get_latest_stats(team, date, stats_df):
            team_stats = stats_df.loc[(slice(None), team), :]
            team_stats_before_date = team_stats[team_stats.index.get_level_values('Date') <= date]
            return team_stats_before_date.iloc[-1] if not team_stats_before_date.empty else None

        # Add home team stats columns
        for column in self.all_team_stats.columns:
            enhanced_schedule[f'Home_{column}'] = enhanced_schedule.apply(
                lambda row: get_latest_stats(row['Home'], row['Date'], self.all_team_stats)[column] 
                if get_latest_stats(row['Home'], row['Date'], self.all_team_stats) is not None 
                else None, 
                axis=1
            )

        # Add away team stats columns 
        for column in self.all_team_stats.columns:
            enhanced_schedule[f'Away_{column}'] = enhanced_schedule.apply(
                lambda row: get_latest_stats(row['Away'], row['Date'], self.all_team_stats)[column]
                if get_latest_stats(row['Away'], row['Date'], self.all_team_stats) is not None
                else None,
                axis=1
            )
            
        self.enhanced_schedule = enhanced_schedule
        return self.enhanced_schedule

    def load_all(self):
        """Load all data in sequence."""
        if self.load_from_files and not self.load_new:
            try:
                self.load_all_from_files()
                print("Successfully loaded all data from files")
                return
            except FileNotFoundError as e:
                print(f"Error loading from files: {str(e)}")
                print("Falling back to loading from web...")
                self.load_from_files = False

        if self.load_new:
            # Load existing data first
            try:
                self.load_all_from_files()
            except FileNotFoundError:
                print("No existing combined stats file found. Loading all data.")

        # Get up to date schedule
        self.get_nba_team_abbreviations()
        self.get_raw_schedule_and_results()
        self.get_schedule_no_results()
        self.set_up_schedule_and_results()
        
        
        self.load_combined_schedule_and_game_stats(reload=self.reload_all)
        self.split_home_away_games()
        self.calculate_cumulative_team_stats()
        self.enhance_schedule_with_team_stats()
       

        if self.reload_all or self.load_new:
            self.save_all_tables()

    def save_all_tables(self):
        """Save all important DataFrames to CSV files in the specified folder."""
        # Dictionary mapping attributes to filenames
        tables_to_save = {
            'nba_abbreviations': 'nba_team_abbreviations.csv',
            'schedule_and_results': 'schedule_and_results.csv',
            'combined_stats': 'combined_schedule_and_game_stats.csv',
            'home_games_df': 'home_games_stats.csv',
            'away_games_df': 'away_games_stats.csv',
            'all_team_stats': 'all_team_stats.csv',
            'enhanced_schedule': 'enhanced_schedule.csv',
            'schedule_no_results': 'schedule_no_results.csv'
        }
        
        # Save each table if it exists
        for attr, filename in tables_to_save.items():
            df = getattr(self, attr)
            if df is not None:
                try:
                    filepath = os.path.join(self.data_folder, filename)
                    df.to_csv(filepath)
                    print(f"Successfully saved {filepath}")
                except Exception as e:
                    print(f"Error saving {filepath}: {str(e)}")

    def load_all_from_files(self):
        """Load all data from files."""
        try:
            self.nba_abbreviations = pd.read_csv(os.path.join(self.data_folder, 'nba_team_abbreviations.csv'), index_col=0)
            self.schedule_and_results = pd.read_csv(os.path.join(self.data_folder, 'schedule_and_results.csv'), index_col=0)
            self.combined_stats = pd.read_csv(os.path.join(self.data_folder, 'combined_schedule_and_game_stats.csv'), index_col=0)
            self.home_games_df = pd.read_csv(os.path.join(self.data_folder, 'home_games_stats.csv'), index_col=[0,1])
            self.away_games_df = pd.read_csv(os.path.join(self.data_folder, 'away_games_stats.csv'), index_col=[0,1])
            self.all_team_stats = pd.read_csv(os.path.join(self.data_folder, 'all_team_stats.csv'), index_col=[0,1])
            self.enhanced_schedule = pd.read_csv(os.path.join(self.data_folder, 'enhanced_schedule.csv'), index_col=0)
            self.schedule_no_results = pd.read_csv(os.path.join(self.data_folder, 'schedule_no_results.csv'), index_col=0)

            # Print and convert date columns for each table that has them
            date_tables = {
                'schedule_and_results': self.schedule_and_results,
                'combined_stats': self.combined_stats, 
                'home_games_df': self.home_games_df,
                'away_games_df': self.away_games_df,
                'all_team_stats': self.all_team_stats,
                'enhanced_schedule': self.enhanced_schedule,
                'schedule_no_results': self.schedule_no_results
            }

            for table_name, df in date_tables.items():
                if isinstance(df.index, pd.MultiIndex):
                    # Convert the 'Date' level to datetime while preserving the index structure
                    if df.index.get_level_values('Date').dtype == 'int64':
                        date_level = pd.to_datetime(df.index.get_level_values('Date'), format="%Y%m%d")
                    else:
                        date_level = pd.to_datetime(df.index.get_level_values('Date'))
                    team_level = df.index.get_level_values('Team')
                    
                    # Create new MultiIndex with converted dates
                    new_index = pd.MultiIndex.from_arrays([date_level, team_level], names=['Date', 'Team'])
                    df.index = new_index
                    setattr(self, table_name, df)
                else:
                    if 'Date' in df.columns:
                        if df["Date"].dtype == 'int64':
                            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
                        else:
                            df["Date"] = pd.to_datetime(df["Date"])
                        setattr(self, table_name, df)
   


        except FileNotFoundError as e:
            print(f"Error loading from files: {str(e)}")
            raise FileNotFoundError(f"Error loading from files: {str(e)}")

