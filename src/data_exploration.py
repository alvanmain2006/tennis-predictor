import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(filepath='data/raw/all_matches_raw.csv'):
    """Load the collected tennis data"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} matches")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        print("  Run the data collector first!")
        return None

def basic_exploration(df):
    """Basic data exploration"""
    
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nData types:")
    print(df.dtypes.value_counts())
    
    print("\nMemory usage:")
    print(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def analyze_matches_by_surface(df):
    """Analyze match distribution by surface"""
    
    
    if 'surface' not in df.columns:
        print("No surface data available")
        return
    
    surface_counts = df['surface'].value_counts()
    print("\nMatch counts:")
    print(surface_counts)
    
    print("\nPercentages:")
    print((surface_counts / len(df) * 100).round(1))
    
    # Plot
    plt.figure(figsize=(10, 6))
    surface_counts.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.title('Tennis Matches by Surface Type', fontsize=16, fontweight='bold')
    plt.xlabel('Surface', fontsize=12)
    plt.ylabel('Number of Matches', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/raw/matches_by_surface.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved plot: data/raw/matches_by_surface.png")
    plt.close()

def analyze_top_players(df, n=20):
    """Find top players by match count"""
    
    # Count matches for each player (as winner or loser)
    winners = df['winner_name'].value_counts()
    losers = df['loser_name'].value_counts()
    
    total_matches = (winners + losers).fillna(0).sort_values(ascending=False)
    
    print(f"\nTop {n} players:")
    for i, (player, matches) in enumerate(total_matches.head(n).items(), 1):
        wins = winners.get(player, 0)
        losses = losers.get(player, 0)
        win_pct = (wins / matches * 100) if matches > 0 else 0
        print(f"{i:2d}. {player:30s} - {int(matches):4d} matches ({wins:3.0f}W-{losses:3.0f}L, {win_pct:.1f}%)")
    
    return total_matches

def analyze_ranking_distribution(df):
    """Analyze player ranking distribution"""
    # Combine winner and loser rankings
    all_ranks = pd.concat([df['winner_rank'], df['loser_rank']]).dropna()
    
    print(f"\nRanking statistics:")
    print(f"  Mean ranking: {all_ranks.mean():.1f}")
    print(f"  Median ranking: {all_ranks.median():.1f}")
    print(f"  Min ranking: {all_ranks.min():.0f}")
    print(f"  Max ranking: {all_ranks.max():.0f}")
    
    # Plot ranking distribution
    plt.figure(figsize=(12, 6))
    all_ranks[all_ranks <= 200].hist(bins=50, color='#3498db', edgecolor='black')
    plt.title('Distribution of Player Rankings (Top 200)', fontsize=16, fontweight='bold')
    plt.xlabel('Ranking', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/raw/ranking_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved plot: data/raw/ranking_distribution.png")
    plt.close()

def analyze_upsets(df, rank_diff_threshold=50):
    """Analyze upset matches (lower-ranked player beats higher-ranked)"""
    
    
    # Calculate ranking difference
    df_analysis = df.dropna(subset=['winner_rank', 'loser_rank']).copy()
    df_analysis['rank_diff'] = df_analysis['loser_rank'] - df_analysis['winner_rank']
    
    # Upsets are when winner_rank > loser_rank (higher number = lower rank)
    upsets = df_analysis[df_analysis['rank_diff'] < -rank_diff_threshold]
    
    print(f"\nTotal matches with ranking data: {len(df_analysis):,}")
    print(f"Major upsets (rank diff > {rank_diff_threshold}): {len(upsets):,}")
    print(f"Upset rate: {len(upsets)/len(df_analysis)*100:.2f}%")
    
    if len(upsets) > 0:
        print(f"\nBiggest upsets:")
        biggest_upsets = upsets.nsmallest(10, 'rank_diff')[
            ['tourney_name', 'winner_name', 'winner_rank', 'loser_name', 'loser_rank', 'rank_diff']
        ]
        print(biggest_upsets.to_string(index=False))

def analyze_by_tournament_level(df):
    """Analyze matches by tournament level"""
   
    
    if 'tourney_level' not in df.columns:
        print("No tournament level data available")
        return
    
    level_mapping = {
        'G': 'Grand Slam',
        'M': 'Masters 1000',
        'A': 'ATP 500',
        'D': 'Davis Cup',
        'F': 'Tour Finals',
        'P': 'Premier',
        'PM': 'Premier Mandatory'
    }
    
    level_counts = df['tourney_level'].value_counts()
    
    print("\nMatch counts by level:")
    for level, count in level_counts.items():
        level_name = level_mapping.get(level, level)
        print(f"  {level_name:20s}: {count:5,} matches")

def check_data_quality(df):
    """Check data quality and missing values"""
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    
    # Only show columns with missing data
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("✓ No missing values found!")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")

def create_quick_summary(df):
    """Create a quick summary report"""
    
    # Convert date
    df_temp = df.copy()
    df_temp['tourney_date'] = pd.to_datetime(df_temp['tourney_date'].astype(str), format='%Y%m%d', errors='coerce')
    
    print(f"\n Total Matches: {len(df):,}")
    print(f" Date Range: {df_temp['tourney_date'].min().strftime('%Y-%m-%d')} to {df_temp['tourney_date'].max().strftime('%Y-%m-%d')}")
    
    if 'tour' in df.columns:
        print(f" Tours: {', '.join(df['tour'].unique())}")
    
    if 'surface' in df.columns:
        print(f"  Surfaces: {', '.join(df['surface'].dropna().unique())}")
    
    unique_players = pd.concat([df['winner_name'], df['loser_name']]).nunique()
    print(f" Unique Players: {unique_players:,}")
    
    if 'tourney_name' in df.columns:
        unique_tournaments = df['tourney_name'].nunique()
        print(f" Unique Tournaments: {unique_tournaments}")


def main():
    # Load data
    df = load_data()
    
    if df is None:
        return
    
    # Run all analyses
    create_quick_summary(df)
    basic_exploration(df)
    check_data_quality(df)
    analyze_matches_by_surface(df)
    analyze_by_tournament_level(df)
    analyze_ranking_distribution(df)
    top_players = analyze_top_players(df, n=20)
    analyze_upsets(df, rank_diff_threshold=50)

    print("\nGenerated files:")
    print("  - data/raw/matches_by_surface.png")
    print("  - data/raw/ranking_distribution.png")
    print("\nNext steps:")
    print("  1. Review the generated visualizations")
    print("  2. Identify any data quality issues")
    print("  3. Move on to data cleaning and processing")
    print()


if __name__ == "__main__":
    main()