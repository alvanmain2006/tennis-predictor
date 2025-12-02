"""
Tennis Match Prediction Model
Train and evaluate machine learning models for tennis match prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import xgboost as xgb
import pickle
import os

class TennisMatchPredictor:
    """Train and evaluate tennis match prediction models"""
    
    def __init__(self, input_file='data/features/match_features.csv',
                 output_dir='models'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_columns = []
        self.models = {}
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def load_data(self):
        """Load featured data"""
        print("="*60)
        print("LOADING FEATURED DATA")
        print("="*60)
        
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"✓ Loaded {len(self.df):,} records")
            print(f"  Columns: {len(self.df.columns)}")
            
            # Convert date
            if 'tourney_date' in self.df.columns:
                self.df['tourney_date'] = pd.to_datetime(self.df['tourney_date'])
            
            return True
        except FileNotFoundError:
            print(f"✗ File not found: {self.input_file}")
            print("  Run feature engineering first!")
            return False
    
    def select_features(self):
        """Select features for modeling"""
        print("\n" + "="*60)
        print("SELECTING FEATURES")
        print("="*60)
        
        # Define feature columns to use
        base_features = [
            'player_rank', 'opponent_rank', 'rank_diff', 
            'is_higher_ranked', 'rank_ratio',
            'player_form', 'player_surface_form',
            'h2h_win_rate', 
            'rest_days', 'matches_last_30_days',
            'ranking_momentum',
            'tournament_win_rate', 'tournament_matches',
            'player_experience'
        ]
        
        # Add surface dummy columns
        surface_cols = [col for col in self.df.columns if col.startswith('surface_')]
        
        # Combine all features
        all_features = base_features + surface_cols
        
        # Only keep features that exist in the dataframe
        self.feature_columns = [col for col in all_features if col in self.df.columns]
        
        print(f"✓ Selected {len(self.feature_columns)} features:")
        for i, feat in enumerate(self.feature_columns, 1):
            print(f"  {i:2d}. {feat}")
        
        # Check for missing values
        missing = self.df[self.feature_columns].isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠ Warning: Missing values detected")
            print(missing[missing > 0])
    
    def prepare_data_splits(self, test_year=2024, val_year=2023):
        """
        Split data by time (crucial for time-series data!)
        Training: Before val_year
        Validation: val_year
        Test: test_year
        """
        print("\n" + "="*60)
        print("PREPARING DATA SPLITS")
        print("="*60)
        
        # Remove rows with missing values in features or outcome
        df_clean = self.df.dropna(subset=self.feature_columns + ['outcome'])
        
        print(f"  Removed {len(self.df) - len(df_clean):,} rows with missing values")
        print(f"  Remaining: {len(df_clean):,} records")
        
        # Create time-based splits
        train_mask = df_clean['year'] < val_year
        val_mask = df_clean['year'] == val_year
        test_mask = df_clean['year'] == test_year
        
        X = df_clean[self.feature_columns]
        y = df_clean['outcome']
        
        self.X_train = X[train_mask]
        self.y_train = y[train_mask]
        
        self.X_val = X[val_mask]
        self.y_val = y[val_mask]
        
        self.X_test = X[test_mask]
        self.y_test = y[test_mask]
        
        print(f"\n✓ Data split by year:")
        print(f"  Training   (< {val_year}): {len(self.X_train):6,} samples")
        print(f"  Validation ({val_year}):     {len(self.X_val):6,} samples")
        print(f"  Test       ({test_year}):     {len(self.X_test):6,} samples")
        
        # Check outcome distribution
        print(f"\n  Outcome distribution:")
        print(f"  Train - Wins: {self.y_train.sum():,} ({self.y_train.mean()*100:.1f}%)")
        print(f"  Val   - Wins: {self.y_val.sum():,} ({self.y_val.mean()*100:.1f}%)")
        print(f"  Test  - Wins: {self.y_test.sum():,} ({self.y_test.mean()*100:.1f}%)")
    
    def train_baseline_model(self):
        """Train baseline logistic regression model"""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL (Logistic Regression)")
        print("="*60)
        
        print("  Training...", end=' ')
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        )
        
        model.fit(self.X_train, self.y_train)
        
        print("✓ Done")
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        
        train_acc = accuracy_score(self.y_train, train_pred)
        val_acc = accuracy_score(self.y_val, val_pred)
        
        print(f"\n  Training Accuracy:   {train_acc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        
        # Store model
        self.models['logistic'] = model
        
        return model
    
    def train_xgboost_model(self):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        print("  Initializing model...")
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20  # Moved here for newer XGBoost versions
        )
        
        print("  Training with early stopping...\n")
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
            verbose=10
        )
        
        print(f"\n✓ Training complete")
        if hasattr(model, 'best_iteration'):
            print(f"  Best iteration: {model.best_iteration}")
        
        # Store model
        self.models['xgboost'] = model
        
        return model
    
    def evaluate_model(self, model_name='xgboost', save_results=True):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print(f"EVALUATING {model_name.upper()} MODEL")
        print("="*60)
        
        model = self.models[model_name]
        
        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        test_pred = model.predict(self.X_test)
        
        train_pred_proba = model.predict_proba(self.X_train)[:, 1]
        val_pred_proba = model.predict_proba(self.X_val)[:, 1]
        test_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Accuracy
        train_acc = accuracy_score(self.y_train, train_pred)
        val_acc = accuracy_score(self.y_val, val_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # AUC
        train_auc = roc_auc_score(self.y_train, train_pred_proba)
        val_auc = roc_auc_score(self.y_val, val_pred_proba)
        test_auc = roc_auc_score(self.y_test, test_pred_proba)
        
        print(f"\n{'Dataset':<12} {'Accuracy':<12} {'AUC':<12}")
        print("-" * 40)
        print(f"{'Training':<12} {train_acc:<12.4f} {train_auc:<12.4f}")
        print(f"{'Validation':<12} {val_acc:<12.4f} {val_auc:<12.4f}")
        print(f"{'Test':<12} {test_acc:<12.4f} {test_auc:<12.4f}")
        
        # Detailed test set evaluation
        print(f"\n" + "="*60)
        print("DETAILED TEST SET RESULTS")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, test_pred, 
                                   target_names=['Loss', 'Win']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, test_pred)
        print("\nConfusion Matrix:")
        print(f"              Predicted Loss  Predicted Win")
        print(f"Actual Loss   {cm[0,0]:14d}  {cm[0,1]:13d}")
        print(f"Actual Win    {cm[1,0]:14d}  {cm[1,1]:13d}")
        
        if save_results:
            self.plot_confusion_matrix(cm, model_name)
            self.plot_roc_curve(model_name)
            self.plot_feature_importance(model, model_name)
        
        return {
            'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
            'train_auc': train_auc, 'val_auc': val_auc, 'test_auc': test_auc
        }
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Loss', 'Win'],
                   yticklabels=['Loss', 'Win'])
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved confusion matrix: results/{model_name}_confusion_matrix.png")
        plt.close()
    
    def plot_roc_curve(self, model_name):
        """Plot ROC curve"""
        model = self.models[model_name]
        
        test_pred_proba = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, test_pred_proba)
        auc = roc_auc_score(self.y_test, test_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name.upper()}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve: results/{model_name}_roc_curve.png")
        plt.close()
    
    def plot_feature_importance(self, model, model_name):
        """Plot feature importance"""
        if model_name == 'xgboost':
            importance = model.feature_importances_
        elif model_name == 'logistic':
            importance = np.abs(model.coef_[0])
        else:
            return
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 15 features
        top_n = min(15, len(importance_df))
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importance_df['importance'].head(top_n))
        plt.yticks(range(top_n), importance_df['feature'].head(top_n))
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {model_name.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'results/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance: results/{model_name}_feature_importance.png")
        plt.close()
        
        # Print top features
        print(f"\nTop 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:<30s}: {row['importance']:.4f}")
    
    def save_model(self, model_name='xgboost'):
        """Save trained model"""
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found in trained models")
            return
        
        model = self.models[model_name]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        model_path = f"{self.output_dir}/{model_name}_model.pkl"
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"\n✓ Model saved to: {model_path}")
            
            # Verify file was created
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / 1024  # KB
                print(f"  File size: {file_size:.1f} KB")
            else:
                print(f"✗ Warning: File not found after saving!")
                
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            return
        
        # Save feature columns
        features_path = f"{self.output_dir}/feature_columns.pkl"
        
        try:
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            print(f"✓ Feature columns saved to: {features_path}")
            
            # Verify file was created
            if os.path.exists(features_path):
                print(f"  Features: {len(self.feature_columns)}")
            else:
                print(f"✗ Warning: Features file not found after saving!")
                
        except Exception as e:
            print(f"✗ Error saving features: {e}")
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        results = []
        
        for model_name in self.models.keys():
            model = self.models[model_name]
            
            test_pred = model.predict(self.X_test)
            test_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            acc = accuracy_score(self.y_test, test_pred)
            auc = roc_auc_score(self.y_test, test_pred_proba)
            
            results.append({
                'Model': model_name,
                'Accuracy': acc,
                'AUC': auc
            })
        
        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        
        print("\nTest Set Performance:")
        print(results_df.to_string(index=False))
        
        best_model = results_df.iloc[0]['Model']
        print(f"\n✓ Best model: {best_model.upper()}")
        
        return results_df
    
    def predict_match(self, player_features, model_name='xgboost'):
        """
        Predict a single match outcome
        
        player_features: dict with feature values
        """
        model = self.models[model_name]
        
        # Create feature vector
        feature_vector = pd.DataFrame([player_features])[self.feature_columns]
        
        # Predict
        prob = model.predict_proba(feature_vector)[0, 1]
        prediction = model.predict(feature_vector)[0]
        
        return {
            'win_probability': prob,
            'prediction': 'Win' if prediction == 1 else 'Loss'
        }
    
    def run_full_pipeline(self):
        """Run complete model training and evaluation pipeline"""
        print("\n")
        print("╔════════════════════════════════════════════════════════╗")
        print("║  TENNIS MATCH PREDICTION MODEL TRAINING              ║")
        print("╚════════════════════════════════════════════════════════╝")
        print()
        
        # Load and prepare data
        if not self.load_data():
            return
        
        self.select_features()
        self.prepare_data_splits(test_year=2024, val_year=2023)
        
        # Train models
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        self.train_baseline_model()
        self.train_xgboost_model()
        
        # Evaluate models
        self.evaluate_model('logistic', save_results=True)
        self.evaluate_model('xgboost', save_results=True)
        
        # Compare
        comparison = self.compare_models()
        
        # Save best model
        best_model = comparison.iloc[0]['Model']
        self.save_model(best_model)
        
        print("\n" + "="*60)
        print("✓ MODEL TRAINING COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  Models:")
        print(f"    - models/{best_model}_model.pkl")
        print(f"    - models/feature_columns.pkl")
        print(f"  Results:")
        print(f"    - results/{best_model}_confusion_matrix.png")
        print(f"    - results/{best_model}_roc_curve.png")
        print(f"    - results/{best_model}_feature_importance.png")
        print("\nNext steps:")
        print("  → Test predictions on new matches")
        print("  → Build tournament simulation")
        print("  → Deploy model for real-time predictions")
        print()


def main():
    """Main execution"""
    predictor = TennisMatchPredictor()
    predictor.run_full_pipeline()


if __name__ == "__main__":
    main()