"""
Tennis Match Prediction Script
Make predictions using trained model
"""

import pickle
import pandas as pd
import numpy as np
import os

class TennisPredictor:
    """Make tennis match predictions"""
    
    def __init__(self, model_path='models/xgboost_model.pkl',
                 features_path='models/feature_columns.pkl'):
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.feature_columns = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and feature columns"""
        if not os.path.exists(self.model_path):
            print(f"✗ Model file not found: {self.model_path}")
            print("  Please train the model first by running: python src/model.py")
            return False
        
        if not os.path.exists(self.features_path):
            print(f"✗ Features file not found: {self.features_path}")
            print("  Please train the model first by running: python src/model.py")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print("✓ Model loaded successfully")
            print(f"  Features: {len(self.feature_columns)}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def predict_match(self, player1_features, player2_features=None):
        """
        Predict match outcome
        
        Args:
            player1_features: dict with player 1's features
            player2_features: dict with player 2's features (optional)
                            If not provided, will use opponent_ fields from player1_features
        
        Returns:
            dict with prediction results
        """
        if self.model is None:
            print("✗ Model not loaded. Cannot make predictions.")
            return None
        
        # If player2_features provided, combine them
        if player2_features is not None:
            combined_features = self._combine_player_features(player1_features, player2_features)
        else:
            combined_features = player1_features
        
        # Create feature vector
        try:
            feature_vector = pd.DataFrame([combined_features])[self.feature_columns]
        except KeyError as e:
            print(f"✗ Missing features: {e}")
            print(f"  Required features: {self.feature_columns}")
            return None
        
        # Make prediction
        prob = self.model.predict_proba(feature_vector)[0, 1]
        prediction = self.model.predict(feature_vector)[0]
        
        return {
            'player1_win_probability': prob,
            'player2_win_probability': 1 - prob,
            'predicted_winner': 'Player 1' if prediction == 1 else 'Player 2',
            'confidence': max(prob, 1 - prob)
        }
    
    def _combine_player_features(self, player1, player2):
        """Combine two player feature dicts into match features"""
        match_features = {
            'player_rank': player1.get('rank', 50),
            'opponent_rank': player2.get('rank', 50),
            'rank_diff': player1.get('rank', 50) - player2.get('rank', 50),
            'is_higher_ranked': 1 if player1.get('rank', 50) < player2.get('rank', 50) else 0,
            'rank_ratio': player1.get('rank', 50) / (player2.get('rank', 50) + 1),
            'player_form': player1.get('form', 0.5),
            'player_surface_form': player1.get('surface_form', 0.5),
            'h2h_win_rate': player1.get('h2h_win_rate', 0.5),
            'rest_days': player1.get('rest_days', 7),
            'matches_last_30_days': player1.get('matches_last_30_days', 3),
            'ranking_momentum': player1.get('ranking_momentum', 0),
            'tournament_win_rate': player1.get('tournament_win_rate', 0.5),
            'tournament_matches': player1.get('tournament_matches', 5),
            'player_experience': player1.get('experience', 100),
        }
        
        # Add surface encoding
        surface = player1.get('surface', 'Hard')
        match_features['surface_Hard'] = 1 if surface == 'Hard' else 0
        match_features['surface_Clay'] = 1 if surface == 'Clay' else 0
        match_features['surface_Grass'] = 1 if surface == 'Grass' else 0
        
        return match_features
    
    def predict_from_names(self, player1_name, player2_name, surface='Hard',
                          player1_rank=None, player2_rank=None):
        """
        Simplified prediction interface using player names
        Uses default/estimated values for other features
        """
        # Create feature dicts with defaults
        player1 = {
            'rank': player1_rank if player1_rank else 10,
            'form': 0.6,  # Default: 60% recent win rate
            'surface_form': 0.6,
            'h2h_win_rate': 0.5,  # Neutral
            'rest_days': 7,
            'matches_last_30_days': 4,
            'ranking_momentum': 0,
            'tournament_win_rate': 0.5,
            'tournament_matches': 5,
            'experience': 100,
            'surface': surface
        }
        
        player2 = {
            'rank': player2_rank if player2_rank else 15,
            'form': 0.6,
            'surface_form': 0.6,
            'h2h_win_rate': 0.5,
            'rest_days': 7,
            'matches_last_30_days': 4,
            'ranking_momentum': 0,
            'tournament_win_rate': 0.5,
            'tournament_matches': 5,
            'experience': 100,
            'surface': surface
        }
        
        result = self.predict_match(player1, player2)
        
        if result:
            print("\n" + "="*60)
            print(f"MATCH PREDICTION: {player1_name} vs {player2_name}")
            print("="*60)
            print(f"Surface: {surface}")
            print(f"\n{player1_name} (Rank {player1['rank']}):")
            print(f"  Win Probability: {result['player1_win_probability']:.1%}")
            print(f"\n{player2_name} (Rank {player2['rank']}):")
            print(f"  Win Probability: {result['player2_win_probability']:.1%}")
            print(f"\nPredicted Winner: {player1_name if result['predicted_winner'] == 'Player 1' else player2_name}")
            print(f"Confidence: {result['confidence']:.1%}")
            print("="*60)
        
        return result


def example_predictions():
    """Example predictions"""
    
    predictor = TennisPredictor()
    
    if predictor.model is None:
        return
    
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║  TENNIS MATCH PREDICTIONS - EXAMPLES                  ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    # Example 1: Simple prediction with just names and ranks
    print("\n--- Example 1: Basic Prediction ---")
    predictor.predict_from_names(
        "Novak Djokovic", 
        "Carlos Alcaraz",
        surface="Hard",
        player1_rank=1,
        player2_rank=2
    )
    
    # Example 2: Detailed prediction with all features
    print("\n--- Example 2: Detailed Prediction ---")
    
    # Simulate Djokovic in great form
    djokovic_features = {
        'player_rank': 1,
        'opponent_rank': 3,
        'rank_diff': -2,
        'is_higher_ranked': 1,
        'rank_ratio': 0.33,
        'player_form': 0.85,  # 85% win rate recently
        'player_surface_form': 0.88,  # Great on hard courts
        'h2h_win_rate': 0.65,  # 65% against this opponent
        'rest_days': 3,
        'matches_last_30_days': 5,
        'ranking_momentum': 0,
        'tournament_win_rate': 0.80,  # Historically good at this tournament
        'tournament_matches': 15,
        'player_experience': 500,
        'surface_Hard': 1,
        'surface_Clay': 0,
        'surface_Grass': 0
    }
    
    result = predictor.predict_match(djokovic_features)
    
    if result:
        print("\n" + "="*60)
        print("DETAILED MATCH PREDICTION")
        print("="*60)
        print(f"Win Probability: {result['player1_win_probability']:.1%}")
        print(f"Predicted Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("="*60)
    
    # Example 3: Clay court specialist vs hard court player
    print("\n--- Example 3: Surface Matchup ---")
    predictor.predict_from_names(
        "Rafael Nadal",
        "Daniil Medvedev", 
        surface="Clay",
        player1_rank=4,
        player2_rank=3
    )


def interactive_prediction():
    """Interactive prediction mode"""
    
    predictor = TennisPredictor()
    
    if predictor.model is None:
        return
    
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║  INTERACTIVE MATCH PREDICTION                         ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    print("\nEnter match details:")
    
    player1_name = input("Player 1 name: ").strip()
    player1_rank = int(input("Player 1 ranking: "))
    
    player2_name = input("Player 2 name: ").strip()
    player2_rank = int(input("Player 2 ranking: "))
    
    print("\nSurface options: Hard, Clay, Grass")
    surface = input("Surface: ").strip().capitalize()
    
    if surface not in ['Hard', 'Clay', 'Grass']:
        surface = 'Hard'
        print(f"Invalid surface. Using: {surface}")
    
    result = predictor.predict_from_names(
        player1_name, player2_name, surface,
        player1_rank, player2_rank
    )


def main():
    """Main execution"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        interactive_prediction()
    else:
        example_predictions()
        
        print("\n\nTo use interactive mode, run:")
        print("  python src/prediction.py interactive")


if __name__ == "__main__":
    main()