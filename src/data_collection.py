import pandas as pd
import time
from datetime import datetime
import os
import ssl

# Fix for SSL certificate issues on Mac
ssl._create_default_https_context = ssl._create_unverified_context

class TennisDataCollector:
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir #self is like this
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Created directory: {self.output_dir}")
    
    def download_sackmann_data(self, years=None, tours=['atp', 'wta']):
        """
        Download data from Jeff Sackmann's GitHub repository
        This is the most comprehensive free tennis dataset
        
        Args:
            years: List of years to download (default: 2021-2024)
            tours: List of tours ['atp', 'wta']
        """
        if years is None:
            years = [2021, 2022, 2023, 2024]
        
        base_urls = {
            'atp': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/',
            'wta': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/'
        }
        
        all_matches = [] #list of multiple datasets
        
        for tour in tours:
            # print(f"Downloading {tour} data...")
            base_url = base_urls[tour]
            
            for year in years:
                url = f"{base_url}{tour}_matches_{year}.csv"
                
                try:
                    # print(f"  Fetching {year}...", end=' ')
                    df = pd.read_csv(url)
                    df['tour'] = tour.upper()
                    df['year'] = year
                    all_matches.append(df)
                    print(f"✓ {len(df)} matches")
                    time.sleep(0.5)  # Be nice to GitHub's servers
                    
                except Exception as e:
                    print(f"✗ Failed: {e}")
        
        if not all_matches:
            print("No data downloaded!")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_matches, ignore_index=False)
        
        # Save to file
        output_file = f"{self.output_dir}/all_matches_raw.csv"
        combined_df.to_csv(output_file, index=True) #to csv takes in path of the file, no index
        
        print(f"\n Total matches collected: {len(combined_df):,}")
        print(f" Saved to: {output_file}")
        
        return combined_df
    
    def download_player_rankings(self, tours=['atp', 'wta']):
        """
        Download historical player rankings
        Useful for getting ranking data for all players
        """
        base_urls = {
            'atp': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv',
            'wta': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv'
        }
        
        all_rankings = []
        
        for tour in tours:
            url = base_urls[tour]
            
            try:
                print(f"Fetching {tour.upper()} rankings...", end=' ')
                df = pd.read_csv(url)
                df['tour'] = tour.upper()
                all_rankings.append(df)
                print(f"✓ {len(df):,} ranking records")
                
            except Exception as e:
                print(f" Failed: {e}")
        
        if not all_rankings:
            return None
        
        combined_rankings = pd.concat(all_rankings, ignore_index=True)
        
        output_file = f"{self.output_dir}/player_rankings.csv"
        combined_rankings.to_csv(output_file, index=False)
        
        print(f"✓ Saved to: {output_file}")
        
        return combined_rankings
    
    def get_data_summary(self, df):
        """
        Print summary statistics about the collected data
        """
        if df is None or df.empty:
            print("No data to summarize")
            return
        
        print("\n" + "="*50)
        print("DATA COLLECTION SUMMARY")
        
        print(f"\nTotal Matches: {len(df):,}")
        
        if 'tour' in df.columns:
            print("\nMatches by Tour:")
            print(df['tour'].value_counts())
        
        if 'year' in df.columns:
            print("\nMatches by Year:")
            print(df['year'].value_counts().sort_index())
        
        if 'surface' in df.columns:
            print("\nMatches by Surface:")
            print(df['surface'].value_counts())
        
        if 'tourney_level' in df.columns:
            print("\nMatches by Tournament Level:")
            print(df['tourney_level'].value_counts())
        
        # Date range
        if 'tourney_date' in df.columns:
            df_temp = df.copy()
            df_temp['tourney_date'] = pd.to_datetime(df_temp['tourney_date'].astype(str), format='%Y%m%d', errors='coerce')
            print(f"\nDate Range: {df_temp['tourney_date'].min()} to {df_temp['tourney_date'].max()}")
        
        # Check for missing data
        print("\nMissing Data:")
        important_cols = ['winner_name', 'loser_name', 'winner_rank', 'loser_rank', 'surface', 'score']
        for col in important_cols:
            if col in df.columns:
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                print(f"{col}: {missing_pct:.1f}% missing")
        
        print("\n" + "="*50)
    
    def show_sample_data(self, df, n=5):
        """Show sample matches"""
        if df is None or df.empty:
            return
        
        print("\n=== SAMPLE MATCHES ===\n")
        
        sample = df.head(n)
        
        for idx, row in sample.iterrows():
            winner = row.get('winner_name', 'Unknown')
            loser = row.get('loser_name', 'Unknown')
            score = row.get('score', 'N/A')
            surface = row.get('surface', 'N/A')
            tourney = row.get('tourney_name', 'N/A')
            
            print(f"{winner} def. {loser}")
            print(f"  Score: {score} | Surface: {surface}")
            print(f"  Tournament: {tourney}\n")
    
    def export_data_info(self, df):
        """Export information about the dataset columns"""
        if df is None or df.empty:
            return
        
        info_file = f"{self.output_dir}/data_columns_info.txt"
        
        with open(info_file, 'w') as f:
            f.write("TENNIS DATASET COLUMN INFORMATION\n")
            f.write("="*50 + "\n\n")
            
            f.write("Available Columns:\n")
            for col in df.columns:
                f.write(f"  - {col}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("KEY COLUMNS EXPLANATION:\n")
            f.write("="*50 + "\n\n")
            
            explanations = {
                'tourney_date': 'Date of tournament (YYYYMMDD format)',
                'tourney_name': 'Name of the tournament',
                'surface': 'Court surface (Hard, Clay, Grass)',
                'tourney_level': 'Level (G=Grand Slam, M=Masters, A=ATP500, etc)',
                'winner_name': 'Name of match winner',
                'loser_name': 'Name of match loser',
                'winner_rank': 'ATP/WTA ranking of winner',
                'loser_rank': 'ATP/WTA ranking of loser',
                'score': 'Match score',
                'best_of': 'Best of 3 or 5 sets',
                'w_ace': 'Winner aces',
                'w_df': 'Winner double faults',
                'w_svpt': 'Winner service points',
                'w_1stIn': 'Winner first serves in',
                'w_1stWon': 'Winner first serve points won',
                'w_2ndWon': 'Winner second serve points won',
                'l_ace': 'Loser aces (same pattern for other l_ stats)'
            }
            
            for col, explanation in explanations.items():
                if col in df.columns:
                    f.write(f"{col}:\n  {explanation}\n\n")
        
        print(f"\n✓ Column information saved to: {info_file}")


def main():
    """Main execution function"""

    # Initialize collector
    collector = TennisDataCollector()
    
    # Download match data for last 4 years
    # print("\nStep 1: Downloading match data...")
    matches_df = collector.download_sackmann_data(
        years=[2021, 2022, 2023, 2024],
        tours=['atp', 'wta']
    )
    
    # Download player rankings
    print("\nStep 2: Downloading player rankings...")
    rankings_df = collector.download_player_rankings(tours=['atp', 'wta'])
    
    # Show summary
    if matches_df is not None:
        collector.get_data_summary(matches_df)
        collector.show_sample_data(matches_df)
        collector.export_data_info(matches_df)
    
    
    print("✓ DATA COLLECTION COMPLETE!")
    print("\nNext steps:")
    print("1. Check data/raw/ folder for the downloaded CSV files")
    print("2. Run data_exploration.py to explore the data")
    print("3. Continue to data cleaning and processing")
    print("\n")


if __name__ == "__main__":
    main()