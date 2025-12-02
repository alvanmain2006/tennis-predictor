"""
Tennis Data Cleaning and Processing
Cleans raw match data and prepares it for feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class TennisDataProcessor:
    """Clean and process tennis match data"""
    
    def __init__(self, input_file='data/raw/all_matches_raw.csv', 
                 output_dir='data/processed'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Load raw data"""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"✓ Loaded {len(self.df):,} matches")
            print(f"  Columns: {len(self.df.columns)}")
            return True
        except FileNotFoundError:
            print(f"✗ File not found: {self.input_file}")
            print("  Run the data collector first!")
            return False
    
    def show_initial_stats(self):
        """Show statistics before cleaning"""
        print("\n" + "="*60)
        print("INITIAL DATA STATISTICS")
        print("="*60)
        
        print(f"\nTotal matches: {len(self.df):,}")
        print(f"\nDate range:")
        
        if 'tourney_date' in self.df.columns:
            dates = pd.to_datetime(self.df['tourney_date'].astype(str), 
                                  format='%Y%m%d', errors='coerce')
            print(f"  From: {dates.min()}")
            print(f"  To: {dates.max()}")
        
        print(f"\nMissing data summary:")
        critical_cols = ['winner_name', 'loser_name', 'winner_rank', 
                        'loser_rank', 'surface', 'score']
        
        for col in critical_cols:
            if col in self.df.columns:
                missing = self.df[col].isna().sum()
                pct = (missing / len(self.df)) * 100
                print(f"  {col:15s}: {missing:6,} ({pct:5.2f}%)")
    
    def clean_dates(self):
        """Clean and convert date columns"""
        print("\n" + "="*60)
        print("CLEANING DATES")
        print("="*60)
        
        if 'tourney_date' not in self.df.columns:
            print("✗ No tourney_date column found")
            return
        
        # Convert to datetime
        self.df['tourney_date'] = pd.to_datetime(
            self.df['tourney_date'].astype(str), 
            format='%Y%m%d', 
            errors='coerce'
        )
        
        # Add useful time features
        self.df['year'] = self.df['tourney_date'].dt.year
        self.df['month'] = self.df['tourney_date'].dt.month
        self.df['day_of_year'] = self.df['tourney_date'].dt.dayofyear
        
        # Remove matches with invalid dates
        invalid_dates = self.df['tourney_date'].isna().sum()
        if invalid_dates > 0:
            print(f"  Removing {invalid_dates} matches with invalid dates")
            self.df = self.df.dropna(subset=['tourney_date'])
        
        print(f"✓ Converted dates to datetime format")
        print(f"✓ Added year, month, and day_of_year columns")
        print(f"  Remaining matches: {len(self.df):,}")
    
    def clean_player_names(self):
        """Clean and standardize player names"""
        print("\n" + "="*60)
        print("CLEANING PLAYER NAMES")
        print("="*60)
        
        # Remove matches with missing player names
        before = len(self.df)
        self.df = self.df.dropna(subset=['winner_name', 'loser_name'])
        removed = before - len(self.df)
        
        if removed > 0:
            print(f"  Removed {removed} matches with missing player names")
        
        # Strip whitespace
        self.df['winner_name'] = self.df['winner_name'].str.strip()
        self.df['loser_name'] = self.df['loser_name'].str.strip()
        
        print(f"✓ Cleaned player names")
        print(f"  Unique players: {pd.concat([self.df['winner_name'], self.df['loser_name']]).nunique():,}")
    
    def clean_rankings(self):
        """Clean ranking data"""
        print("\n" + "="*60)
        print("CLEANING RANKINGS")
        print("="*60)
        
        # Fill missing rankings with a high number (1000 = unranked)
        for col in ['winner_rank', 'loser_rank']:
            if col in self.df.columns:
                missing_before = self.df[col].isna().sum()
                self.df[col] = self.df[col].fillna(1000)
                print(f"  {col}: Filled {missing_before:,} missing values with 1000")
        
        # Ensure rankings are integers
        self.df['winner_rank'] = self.df['winner_rank'].astype(int)
        self.df['loser_rank'] = self.df['loser_rank'].astype(int)
        
        # Add ranking difference
        self.df['rank_diff'] = self.df['winner_rank'] - self.df['loser_rank']
        
        print(f"✓ Cleaned ranking data")
        print(f"  Average winner rank: {self.df['winner_rank'].mean():.1f}")
        print(f"  Average loser rank: {self.df['loser_rank'].mean():.1f}")
    
    def clean_surface(self):
        """Clean and standardize surface names"""
        print("\n" + "="*60)
        print("CLEANING SURFACE DATA")
        print("="*60)
        
        if 'surface' not in self.df.columns:
            print("✗ No surface column found")
            return
        
        # Remove matches with missing surface
        before = len(self.df)
        self.df = self.df.dropna(subset=['surface'])
        removed = before - len(self.df)
        
        if removed > 0:
            print(f"  Removed {removed} matches with missing surface")
        
        # Standardize surface names
        surface_mapping = {
            'Hard': 'Hard',
            'Clay': 'Clay',
            'Grass': 'Grass',
            'Carpet': 'Hard',  # Carpet is rare, group with Hard
            'hard': 'Hard',
            'clay': 'Clay',
            'grass': 'Grass',
            'carpet': 'Hard'
        }
        
        self.df['surface'] = self.df['surface'].map(
            lambda x: surface_mapping.get(x, x)
        )
        
        print(f"✓ Standardized surface names")
        print(f"\nSurface distribution:")
        print(self.df['surface'].value_counts())
    
    def clean_match_stats(self):
        """Clean match statistics (aces, double faults, etc.)"""
        print("\n" + "="*60)
        print("CLEANING MATCH STATISTICS")
        print("="*60)
        
        # List of statistical columns
        stat_cols = []
        for prefix in ['w_', 'l_']:
            for stat in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 
                        'bpSaved', 'bpFaced']:
                col = f"{prefix}{stat}"
                if col in self.df.columns:
                    stat_cols.append(col)
        
        if not stat_cols:
            print("  No match statistics found in data")
            return
        
        # Count how many matches have complete stats
        complete_stats = self.df[stat_cols].notna().all(axis=1).sum()
        print(f"  Matches with complete stats: {complete_stats:,} ({complete_stats/len(self.df)*100:.1f}%)")
        
        # Fill missing stats with 0 (or you could remove these matches)
        for col in stat_cols:
            missing = self.df[col].isna().sum()
            if missing > 0:
                self.df[col] = self.df[col].fillna(0)
        
        print(f"✓ Cleaned {len(stat_cols)} statistical columns")
    
    def add_match_outcome_indicator(self):
        """Add a simple outcome column (always 1 for winner perspective)"""
        print("\n" + "="*60)
        print("ADDING OUTCOME INDICATORS")
        print("="*60)
        
        # This will be useful later when we create player-perspective records
        self.df['winner_outcome'] = 1
        self.df['loser_outcome'] = 0
        
        print("✓ Added outcome indicators")
    
    def remove_duplicates(self):
        """Remove duplicate matches"""
        print("\n" + "="*60)
        print("REMOVING DUPLICATES")
        print("="*60)
        
        before = len(self.df)
        
        # Consider a match duplicate if same players, date, and tournament
        duplicate_cols = ['tourney_date', 'tourney_name', 'winner_name', 'loser_name']
        self.df = self.df.drop_duplicates(subset=duplicate_cols, keep='first')
        
        removed = before - len(self.df)
        
        if removed > 0:
            print(f"  Removed {removed} duplicate matches")
        else:
            print(f"  No duplicates found")
        
        print(f"✓ Remaining matches: {len(self.df):,}")
    
    def sort_by_date(self):
        """Sort matches chronologically"""
        print("\n" + "="*60)
        print("SORTING DATA")
        print("="*60)
        
        self.df = self.df.sort_values('tourney_date').reset_index(drop=True)
        print("✓ Sorted matches chronologically")
    
    def create_player_perspective_records(self):
        """
        Create records from both winner and loser perspective
        This is crucial for machine learning - each player needs a record
        """
        print("\n" + "="*60)
        print("CREATING PLAYER-PERSPECTIVE RECORDS")
        print("="*60)
        
        print("  Creating winner records...", end=' ')
        
        # Winner records (outcome = 1)
        winner_records = self.df.copy()
        winner_records['player_name'] = winner_records['winner_name']
        winner_records['opponent_name'] = winner_records['loser_name']
        winner_records['player_rank'] = winner_records['winner_rank']
        winner_records['opponent_rank'] = winner_records['loser_rank']
        winner_records['player_seed'] = winner_records.get('winner_seed', None)
        winner_records['opponent_seed'] = winner_records.get('loser_seed', None)
        winner_records['outcome'] = 1
        
        # Copy match stats
        stat_pairs = [
            ('ace', 'ace'), ('df', 'df'), ('svpt', 'svpt'),
            ('1stIn', '1stIn'), ('1stWon', '1stWon'), ('2ndWon', '2ndWon'),
            ('bpSaved', 'bpSaved'), ('bpFaced', 'bpFaced')
        ]
        
        for stat in stat_pairs:
            w_col = f'w_{stat[0]}'
            l_col = f'l_{stat[0]}'
            if w_col in winner_records.columns:
                winner_records[f'player_{stat[1]}'] = winner_records[w_col]
                winner_records[f'opponent_{stat[1]}'] = winner_records[l_col]
        
        print(f"✓ {len(winner_records):,} records")
        
        print("  Creating loser records...", end=' ')
        
        # Loser records (outcome = 0)
        loser_records = self.df.copy()
        loser_records['player_name'] = loser_records['loser_name']
        loser_records['opponent_name'] = loser_records['winner_name']
        loser_records['player_rank'] = loser_records['loser_rank']
        loser_records['opponent_rank'] = loser_records['winner_rank']
        loser_records['player_seed'] = loser_records.get('loser_seed', None)
        loser_records['opponent_seed'] = loser_records.get('winner_seed', None)
        loser_records['outcome'] = 0
        
        # Copy match stats (reversed)
        for stat in stat_pairs:
            w_col = f'w_{stat[0]}'
            l_col = f'l_{stat[0]}'
            if l_col in loser_records.columns:
                loser_records[f'player_{stat[1]}'] = loser_records[l_col]
                loser_records[f'opponent_{stat[1]}'] = loser_records[w_col]
        
        print(f"✓ {len(loser_records):,} records")
        
        # Combine both perspectives
        print("  Combining records...", end=' ')
        all_records = pd.concat([winner_records, loser_records], ignore_index=True)
        
        # Sort by date to maintain chronological order
        all_records = all_records.sort_values('tourney_date').reset_index(drop=True)
        
        print(f"✓ {len(all_records):,} total records")
        
        return all_records
    
    def save_cleaned_data(self, df, filename='matches_cleaned.csv'):
        """Save cleaned data"""
        print("\n" + "="*60)
        print("SAVING CLEANED DATA")
        print("="*60)
        
        output_path = f"{self.output_dir}/{filename}"
        df.to_csv(output_path, index=False)
        
        print(f"✓ Saved to: {output_path}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        return output_path
    
    def generate_cleaning_report(self):
        """Generate a report of the cleaning process"""
        print("\n" + "="*60)
        print("CLEANING REPORT SUMMARY")
        print("="*60)
        
        report = []
        report.append("TENNIS DATA CLEANING REPORT")
        report.append("="*60)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nFinal dataset statistics:")
        report.append(f"  Total matches: {len(self.df):,}")
        report.append(f"  Date range: {self.df['tourney_date'].min()} to {self.df['tourney_date'].max()}")
        report.append(f"  Unique players: {pd.concat([self.df['winner_name'], self.df['loser_name']]).nunique():,}")
        report.append(f"  Unique tournaments: {self.df['tourney_name'].nunique():,}")
        
        if 'tour' in self.df.columns:
            report.append(f"\nMatches by tour:")
            for tour, count in self.df['tour'].value_counts().items():
                report.append(f"  {tour}: {count:,}")
        
        report.append(f"\nMatches by surface:")
        for surface, count in self.df['surface'].value_counts().items():
            report.append(f"  {surface}: {count:,}")
        
        report.append(f"\nMatches by year:")
        for year, count in self.df['year'].value_counts().sort_index().items():
            report.append(f"  {year}: {count:,}")
        
        # Save report
        report_path = f"{self.output_dir}/cleaning_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        print(f"\n✓ Report saved to: {report_path}")
    
    def process_all(self):
        """Run the complete cleaning pipeline"""
        print("\n")
        print("╔════════════════════════════════════════════════════════╗")
        print("║  TENNIS DATA CLEANING & PROCESSING PIPELINE           ║")
        print("╚════════════════════════════════════════════════════════╝")
        print()
        
        # Load data
        if not self.load_data():
            return
        
        # Show initial stats
        self.show_initial_stats()
        
        # Run cleaning steps
        self.clean_dates()
        self.clean_player_names()
        self.clean_rankings()
        self.clean_surface()
        self.clean_match_stats()
        self.add_match_outcome_indicator()
        self.remove_duplicates()
        self.sort_by_date()
        
        # Save match-level cleaned data
        self.save_cleaned_data(self.df, 'matches_cleaned.csv')
        
        # Create player-perspective records
        player_records = self.create_player_perspective_records()
        
        # Save player-perspective data
        self.save_cleaned_data(player_records, 'player_match_records.csv')
        
        # Generate report
        self.generate_cleaning_report()
        
        print("\n" + "="*60)
        print("✓ DATA CLEANING COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  1. {self.output_dir}/matches_cleaned.csv")
        print(f"     → Match-level data (one row per match)")
        print(f"  2. {self.output_dir}/player_match_records.csv")
        print(f"     → Player-perspective data (two rows per match)")
        print(f"  3. {self.output_dir}/cleaning_report.txt")
        print(f"     → Summary report")
        print("\nNext steps:")
        print("  → Run feature engineering to create ML features")
        print("  → Start building your prediction model")
        print()


def main():
    """Main execution"""
    processor = TennisDataProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()