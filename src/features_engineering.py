"""
Tennis Feature Engineering
Creates predictive features from cleaned match data
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import os

class TennisFeatureEngineer:
    """Create features for tennis match prediction"""
    
    def __init__(self, input_file='data/processed/player_match_records.csv',
                 output_dir='data/features'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Load cleaned data"""
        print("="*60)
        print("LOADING CLEANED DATA")
        print("="*60)
        
        try:
            self.df = pd.read_csv(self.input_file)
            self.df['tourney_date'] = pd.to_datetime(self.df['tourney_date'])
            print(f"✓ Loaded {len(self.df):,} player-match records")
            return True
        except FileNotFoundError:
            print(f"✗ File not found: {self.input_file}")
            print("  Run data cleaning first!")
            return False
    
    def calculate_recent_form(self, lookback_matches=10):
        """
        Calculate recent form (win rate over last N matches)
        This is a rolling calculation that only uses past data
        """
        print("\n" + "="*60)
        print(f"CALCULATING RECENT FORM (Last {lookback_matches} matches)")
        print("="*60)
        
        self.df = self.df.sort_values(['player_name', 'tourney_date']).reset_index(drop=True)
        
        # Calculate rolling win rate for each player
        print("  Computing rolling win rates...", end=' ')
        
        form_values = []
        
        for player in self.df['player_name'].unique():
            player_matches = self.df[self.df['player_name'] == player].copy()
            
            # Calculate rolling mean of outcomes
            player_matches['form'] = player_matches['outcome'].rolling(
                window=lookback_matches, 
                min_periods=1
            ).mean().shift(1)  # shift(1) ensures we don't use current match
            
            # Fill NaN for first match with 0.5 (neutral)
            player_matches['form'] = player_matches['form'].fillna(0.5)
            
            form_values.extend(player_matches['form'].values)
        
        self.df['player_form'] = form_values
        
        print(f"✓ Done")
        print(f"  Average form: {self.df['player_form'].mean():.3f}")
        print(f"  Min form: {self.df['player_form'].min():.3f}")
        print(f"  Max form: {self.df['player_form'].max():.3f}")
    
    def calculate_surface_form(self, lookback_matches=20):
        """
        Calculate form on specific surface
        """
        print("\n" + "="*60)
        print(f"CALCULATING SURFACE-SPECIFIC FORM (Last {lookback_matches} matches)")
        print("="*60)
        
        surface_form_values = []
        
        for surface in self.df['surface'].unique():
            print(f"  Processing {surface}...", end=' ')
            
            surface_df = self.df[self.df['surface'] == surface].copy()
            
            for player in surface_df['player_name'].unique():
                player_surface = surface_df[surface_df['player_name'] == player].copy()
                
                # Rolling win rate on this surface
                player_surface['surface_form'] = player_surface['outcome'].rolling(
                    window=lookback_matches,
                    min_periods=1
                ).mean().shift(1)
                
                player_surface['surface_form'] = player_surface['surface_form'].fillna(0.5)
                
                surface_form_values.extend(player_surface['surface_form'].values)
            
            print(f"✓")
        
        self.df['player_surface_form'] = surface_form_values
        
        print(f"✓ Surface form calculated")
        print(f"  Average: {self.df['player_surface_form'].mean():.3f}")
    
    def calculate_head_to_head(self):
        """
        Calculate head-to-head record between players
        """
        print("\n" + "="*60)
        print("CALCULATING HEAD-TO-HEAD RECORDS")
        print("="*60)
        
        h2h_wins = []
        h2h_total = []
        h2h_win_rate = []
        
        print("  Processing head-to-head...", end=' ')
        
        for idx, row in self.df.iterrows():
            player = row['player_name']
            opponent = row['opponent_name']
            date = row['tourney_date']
            
            # Find all previous matches between these players
            previous_h2h = self.df[
                (self.df['player_name'] == player) &
                (self.df['opponent_name'] == opponent) &
                (self.df['tourney_date'] < date)
            ]
            
            if len(previous_h2h) > 0:
                wins = previous_h2h['outcome'].sum()
                total = len(previous_h2h)
                win_rate = wins / total
            else:
                wins = 0
                total = 0
                win_rate = 0.5  # Neutral if no history
            
            h2h_wins.append(wins)
            h2h_total.append(total)
            h2h_win_rate.append(win_rate)
            
            if idx % 5000 == 0:
                print(f"{idx//1000}k...", end='')
        
        self.df['h2h_wins'] = h2h_wins
        self.df['h2h_total'] = h2h_total
        self.df['h2h_win_rate'] = h2h_win_rate
        
        print(f" ✓ Done")
        print(f"  Matches with H2H history: {(self.df['h2h_total'] > 0).sum():,}")
        print(f"  Average H2H matches: {self.df['h2h_total'].mean():.2f}")
    
    def calculate_ranking_momentum(self, window_days=90):
        """
        Calculate how rankings are changing (momentum)
        """
        print("\n" + "="*60)
        print(f"CALCULATING RANKING MOMENTUM ({window_days} days)")
        print("="*60)
        
        ranking_change = []
        
        print("  Computing ranking trends...", end=' ')
        
        for player in self.df['player_name'].unique():
            player_matches = self.df[self.df['player_name'] == player].copy()
            player_matches = player_matches.sort_values('tourney_date')
            
            for idx, row in player_matches.iterrows():
                current_date = row['tourney_date']
                current_rank = row['player_rank']
                
                # Find rank from window_days ago
                past_date = current_date - timedelta(days=window_days)
                past_matches = player_matches[
                    (player_matches['tourney_date'] >= past_date) &
                    (player_matches['tourney_date'] < current_date)
                ]
                
                if len(past_matches) > 0:
                    past_rank = past_matches.iloc[0]['player_rank']
                    # Negative change = improvement (lower rank number is better)
                    change = past_rank - current_rank
                else:
                    change = 0
                
                ranking_change.append(change)
        
        self.df['ranking_momentum'] = ranking_change
        
        print(f"✓ Done")
        print(f"  Average momentum: {self.df['ranking_momentum'].mean():.1f}")
    
    def calculate_rest_days(self):
        """
        Calculate days since last match (fatigue indicator)
        """
        print("\n" + "="*60)
        print("CALCULATING REST DAYS")
        print("="*60)
        
        rest_days = []
        
        print("  Computing rest periods...", end=' ')
        
        for player in self.df['player_name'].unique():
            player_matches = self.df[self.df['player_name'] == player].copy()
            player_matches = player_matches.sort_values('tourney_date')
            
            for idx, row in player_matches.iterrows():
                current_date = row['tourney_date']
                
                # Find previous match
                previous = player_matches[
                    player_matches['tourney_date'] < current_date
                ]
                
                if len(previous) > 0:
                    last_date = previous.iloc[-1]['tourney_date']
                    days = (current_date - last_date).days
                else:
                    days = 30  # Default for first match
                
                rest_days.append(days)
        
        self.df['rest_days'] = rest_days
        
        print(f"✓ Done")
        print(f"  Average rest: {self.df['rest_days'].mean():.1f} days")
        print(f"  Median rest: {self.df['rest_days'].median():.1f} days")
    
    def calculate_matches_played_last_30_days(self):
        """
        Count matches in last 30 days (activity level)
        """
        print("\n" + "="*60)
        print("CALCULATING RECENT MATCH COUNT")
        print("="*60)
        
        recent_matches = []
        
        print("  Counting recent matches...", end=' ')
        
        for idx, row in self.df.iterrows():
            player = row['player_name']
            current_date = row['tourney_date']
            
            # Count matches in last 30 days (excluding current)
            past_30_days = self.df[
                (self.df['player_name'] == player) &
                (self.df['tourney_date'] >= current_date - timedelta(days=30)) &
                (self.df['tourney_date'] < current_date)
            ]
            
            recent_matches.append(len(past_30_days))
        
        self.df['matches_last_30_days'] = recent_matches
        
        print(f"✓ Done")
        print(f"  Average matches/30 days: {self.df['matches_last_30_days'].mean():.2f}")
    
    def calculate_tournament_history(self):
        """
        Calculate performance at specific tournament
        """
        print("\n" + "="*60)
        print("CALCULATING TOURNAMENT-SPECIFIC HISTORY")
        print("="*60)
        
        tournament_win_rate = []
        tournament_matches = []
        
        print("  Processing tournament history...", end=' ')
        
        for idx, row in self.df.iterrows():
            player = row['player_name']
            tournament = row['tourney_name']
            date = row['tourney_date']
            
            # Find all previous matches at this tournament
            prev_tournament = self.df[
                (self.df['player_name'] == player) &
                (self.df['tourney_name'] == tournament) &
                (self.df['tourney_date'] < date)
            ]
            
            if len(prev_tournament) > 0:
                wins = prev_tournament['outcome'].sum()
                total = len(prev_tournament)
                win_rate = wins / total
            else:
                total = 0
                win_rate = 0.5
            
            tournament_matches.append(total)
            tournament_win_rate.append(win_rate)
            
            if idx % 5000 == 0:
                print(f"{idx//1000}k...", end='')
        
        self.df['tournament_matches'] = tournament_matches
        self.df['tournament_win_rate'] = tournament_win_rate
        
        print(f" ✓ Done")
        print(f"  Players with tournament history: {(self.df['tournament_matches'] > 0).sum():,}")
    
    def add_basic_derived_features(self):
        """
        Add simple calculated features
        """
        print("\n" + "="*60)
        print("ADDING DERIVED FEATURES")
        print("="*60)
        
        # Ranking features
        self.df['rank_diff'] = self.df['player_rank'] - self.df['opponent_rank']
        self.df['is_higher_ranked'] = (self.df['player_rank'] < self.df['opponent_rank']).astype(int)
        
        # Log of rank (helps with ML)
        self.df['player_rank_log'] = np.log1p(self.df['player_rank'])
        self.df['opponent_rank_log'] = np.log1p(self.df['opponent_rank'])
        
        # Rank ratio
        self.df['rank_ratio'] = self.df['player_rank'] / (self.df['opponent_rank'] + 1)
        
        print("✓ Added ranking-based features")
        
        # Time features
        if 'month' in self.df.columns:
            # Season indicators
            self.df['is_hard_court_season'] = self.df['month'].isin([1, 2, 3, 8, 9, 10, 11]).astype(int)
            self.df['is_clay_season'] = self.df['month'].isin([4, 5, 6]).astype(int)
            self.df['is_grass_season'] = self.df['month'].isin([6, 7]).astype(int)
            
            print("✓ Added time-based features")
        
        # Experience indicator (based on matches played)
        player_match_counts = self.df.groupby('player_name').cumcount()
        self.df['player_experience'] = player_match_counts
        
        print("✓ Added experience features")
    
    def encode_categorical_features(self):
        """
        One-hot encode categorical variables
        """
        print("\n" + "="*60)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*60)
        
        # Encode surface
        if 'surface' in self.df.columns:
            surface_dummies = pd.get_dummies(self.df['surface'], prefix='surface')
            self.df = pd.concat([self.df, surface_dummies], axis=1)
            print(f"✓ Encoded surface: {list(surface_dummies.columns)}")
        
        # Encode tournament level if available
        if 'tourney_level' in self.df.columns:
            level_dummies = pd.get_dummies(self.df['tourney_level'], prefix='level')
            self.df = pd.concat([self.df, level_dummies], axis=1)
            print(f"✓ Encoded tournament level")
    
    def create_feature_summary(self):
        """
        Show summary of all created features
        """
        print("\n" + "="*60)
        print("FEATURE SUMMARY")
        print("="*60)
        
        feature_categories = {
            'Ranking Features': [
                'player_rank', 'opponent_rank', 'rank_diff', 
                'is_higher_ranked', 'rank_ratio', 'ranking_momentum'
            ],
            'Form Features': [
                'player_form', 'player_surface_form'
            ],
            'Head-to-Head': [
                'h2h_wins', 'h2h_total', 'h2h_win_rate'
            ],
            'Fatigue/Activity': [
                'rest_days', 'matches_last_30_days'
            ],
            'Tournament History': [
                'tournament_matches', 'tournament_win_rate'
            ],
            'Surface': [col for col in self.df.columns if col.startswith('surface_')],
        }
        
        print("\nFeatures by category:")
        total_features = 0
        
        for category, features in feature_categories.items():
            available = [f for f in features if f in self.df.columns]
            if available:
                print(f"\n{category}:")
                for feature in available:
                    print(f"  - {feature}")
                total_features += len(available)
        
        print(f"\n✓ Total features created: {total_features}")
        print(f"✓ Total columns in dataset: {len(self.df.columns)}")
    
    def save_featured_data(self):
        """
        Save data with features
        """
        print("\n" + "="*60)
        print("SAVING FEATURED DATA")
        print("="*60)
        
        output_path = f"{self.output_dir}/match_features.csv"
        self.df.to_csv(output_path, index=False)
        
        print(f"✓ Saved to: {output_path}")
        print(f"  Rows: {len(self.df):,}")
        print(f"  Columns: {len(self.df.columns)}")
        
        # Save feature list
        feature_list_path = f"{self.output_dir}/feature_list.txt"
        with open(feature_list_path, 'w') as f:
            f.write("TENNIS MATCH PREDICTION FEATURES\n")
            f.write("="*60 + "\n\n")
            f.write("All columns in the dataset:\n\n")
            for col in sorted(self.df.columns):
                f.write(f"  - {col}\n")
        
        print(f"✓ Feature list saved to: {feature_list_path}")
    
    def show_sample_features(self, n=5):
        """
        Display sample records with features
        """
        print("\n" + "="*60)
        print("SAMPLE FEATURED RECORDS")
        print("="*60)
        
        important_cols = [
            'player_name', 'opponent_name', 'surface', 
            'player_rank', 'opponent_rank', 'player_form',
            'player_surface_form', 'h2h_win_rate', 'outcome'
        ]
        
        display_cols = [col for col in important_cols if col in self.df.columns]
        
        sample = self.df[display_cols].head(n)
        
        print("\n", sample.to_string(index=False))
    
    def engineer_all_features(self):
        """
        Run complete feature engineering pipeline
        """
        print("\n")
        print("╔════════════════════════════════════════════════════════╗")
        print("║  TENNIS FEATURE ENGINEERING PIPELINE                  ║")
        print("╚════════════════════════════════════════════════════════╝")
        print()
        
        # Load data
        if not self.load_data():
            return
        
        print(f"\nStarting feature engineering on {len(self.df):,} records...")
        print("This may take a few minutes...\n")
        
        # Create features
        self.calculate_recent_form(lookback_matches=10)
        self.calculate_surface_form(lookback_matches=20)
        self.calculate_head_to_head()
        self.calculate_ranking_momentum(window_days=90)
        self.calculate_rest_days()
        self.calculate_matches_played_last_30_days()
        self.calculate_tournament_history()
        self.add_basic_derived_features()
        self.encode_categorical_features()
        
        # Show results
        self.create_feature_summary()
        self.show_sample_features(n=5)
        
        # Save
        self.save_featured_data()
        
        print("\n" + "="*60)
        print("✓ FEATURE ENGINEERING COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  1. {self.output_dir}/match_features.csv")
        print(f"     → Ready for machine learning!")
        print(f"  2. {self.output_dir}/feature_list.txt")
        print(f"     → List of all features")
        print("\nNext steps:")
        print("  → Build and train your prediction model")
        print("  → Start with simple models (logistic regression)")
        print("  → Then try XGBoost for better performance")
        print()


def main():
    """Main execution"""
    engineer = TennisFeatureEngineer()
    engineer.engineer_all_features()


if __name__ == "__main__":
    main()