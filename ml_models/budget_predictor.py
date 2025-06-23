import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

from data_processor import FinanceDataProcessor

class BudgetPredictor:
    """
    Advanced ML model for predicting future budget needs.
    
    This class analyzes historical spending patterns to predict
    future budget requirements by category and time period.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the budget predictor
        
        Args:
            model_type (str): Type of ML model to use
                            Options: 'linear_regression', 'random_forest'
        """
        self.model_type = model_type
        self.models = {}  # Store separate models for each category
        self.scalers = {}  # Store scalers for each category
        self.data_processor = FinanceDataProcessor()
        self.categories = self.data_processor.categories
        self.is_trained = False
        
        # Choose the ML algorithm
        if model_type == 'linear_regression':
            self.base_model = LinearRegression()
        elif model_type == 'random_forest':
            self.base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def generate_sample_data(self, months=24):
        """
        Generate realistic sample transaction data for training
        
        Args:
            months (int): Number of months of data to generate
            
        Returns:
            pd.DataFrame: Sample transaction data
        """
        np.random.seed(42)  # For reproducible results
        
        # Define spending patterns by category (monthly base amounts and variations)
        category_patterns = {
            'Food & Dining': {'base': 300, 'variation': 100, 'trend': 1.02},
            'Groceries': {'base': 400, 'variation': 80, 'trend': 1.01},
            'Transportation': {'base': 200, 'variation': 60, 'trend': 1.03},
            'Shopping': {'base': 250, 'variation': 150, 'trend': 1.01},
            'Bills & Utilities': {'base': 350, 'variation': 50, 'trend': 1.02},
            'Entertainment': {'base': 150, 'variation': 80, 'trend': 1.01},
            'Healthcare': {'base': 200, 'variation': 100, 'trend': 1.02},
            'Gas': {'base': 120, 'variation': 40, 'trend': 1.03},
            'Other': {'base': 100, 'variation': 50, 'trend': 1.01}
        }
        
        transactions = []
        start_date = datetime.now() - timedelta(days=months * 30)
        
        for month in range(months):
            month_start = start_date + timedelta(days=month * 30)
            
            for category, pattern in category_patterns.items():
                # Calculate monthly spending with trend
                base_amount = pattern['base'] * (pattern['trend'] ** month)
                
                # Add seasonal variation
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
                
                # Add random variation
                monthly_amount = base_amount * seasonal_factor * np.random.normal(1, 0.2)
                monthly_amount = max(monthly_amount, pattern['base'] * 0.3)  # Minimum spending
                
                # Generate individual transactions for the month
                num_transactions = np.random.poisson(8)  # Average 8 transactions per month
                
                for _ in range(num_transactions):
                    # Random day in the month
                    day_offset = np.random.randint(0, 30)
                    transaction_date = month_start + timedelta(days=day_offset)
                    
                    # Random transaction amount (part of monthly total)
                    amount = monthly_amount / num_transactions * np.random.uniform(0.5, 2.0)
                    
                    transactions.append({
                        'date': transaction_date,
                        'amount': amount,
                        'category': category,
                        'description': f'{category} purchase'
                    })
        
        df = pd.DataFrame(transactions)
        df = self.data_processor.prepare_transaction_data(df.to_dict('records'))
        
        print(f"Generated {len(df)} sample transactions over {months} months")
        return df
    
    def prepare_features(self, df, category, lookback_months=6):
        """
        Prepare features for budget prediction
        
        Args:
            df (pd.DataFrame): Transaction data
            category (str): Category to predict for
            lookback_months (int): Number of months to look back for features
            
        Returns:
            tuple: (X, y, dates) features, targets, and dates
        """
        # Filter by category
        category_data = df[df['category'] == category].copy()
        
        if len(category_data) == 0:
            return np.array([]), np.array([]), []
        
        # Group by month
        monthly_data = category_data.groupby([
            category_data['date'].dt.year,
            category_data['date'].dt.month
        ]).agg({
            'amount_abs': ['sum', 'count', 'mean'],
            'day_of_week': 'mean',
            'day_of_month': 'mean'
        }).round(2)
        
        # Flatten column names
        monthly_data.columns = ['total_spent', 'transaction_count', 'avg_amount', 'avg_day_of_week', 'avg_day_of_month']
        
        if len(monthly_data) < lookback_months + 1:
            return np.array([]), np.array([]), []
        
        # Create features and targets
        X, y, dates = [], [], []
        
        for i in range(lookback_months, len(monthly_data)):
            # Features: previous months data
            feature_vector = []
            
            for j in range(lookback_months):
                month_idx = i - lookback_months + j
                month_data = monthly_data.iloc[month_idx]
                
                feature_vector.extend([
                    month_data['total_spent'],
                    month_data['transaction_count'],
                    month_data['avg_amount'],
                    month_data['avg_day_of_week'],
                    month_data['avg_day_of_month']
                ])
            
            # Add time-based features
            current_date = monthly_data.index[i]
            month = current_date[1]  # Month number
            
            # Seasonal features
            feature_vector.extend([
                np.sin(2 * np.pi * month / 12),  # Seasonal sine
                np.cos(2 * np.pi * month / 12),  # Seasonal cosine
                month  # Month number
            ])
            
            # Trend features
            recent_avg = monthly_data.iloc[i-3:i]['total_spent'].mean() if i >= 3 else monthly_data.iloc[0:i]['total_spent'].mean()
            feature_vector.append(recent_avg)
            
            X.append(feature_vector)
            y.append(monthly_data.iloc[i]['total_spent'])
            dates.append(current_date)
        
        return np.array(X), np.array(y), dates
    
    def train_model(self, transaction_data=None):
        """
        Train budget prediction models for all categories
        
        Args:
            transaction_data (pd.DataFrame, optional): Custom transaction data
                                                     If None, generates sample data
        
        Returns:
            dict: Training results and metrics
        """
        print(f"Training {self.model_type} budget predictor...")
        
        # Get or generate training data
        if transaction_data is None:
            transaction_data = self.generate_sample_data(months=24)
        
        if transaction_data.empty:
            raise ValueError("No transaction data available")
        
        results = {}
        
        # Train separate model for each category
        for category in self.categories:
            print(f"Training model for {category}...")
            
            # Prepare features for this category
            X, y, dates = self.prepare_features(transaction_data, category)
            
            if len(X) == 0:
                print(f"  No sufficient data for {category}")
                continue
            
            if len(X) < 3:
                print(f"  Not enough data for {category} (need at least 3 months)")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create and train model
            if self.model_type == 'linear_regression':
                model = LinearRegression()
            else:  # random_forest
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_scaled, y)
            
            # Evaluate model
            y_pred = model.predict(X_scaled)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            # Store model and scaler
            self.models[category] = model
            self.scalers[category] = scaler
            
            results[category] = {
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'training_samples': len(X),
                'avg_prediction': y_pred.mean(),
                'avg_actual': y.mean()
            }
            
            print(f"  MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.3f}")
        
        self.is_trained = True
        print(f"Budget predictor trained for {len(self.models)} categories")
        
        return results
    
    def predict_next_month(self, transaction_data, category, confidence_interval=False):
        """
        Predict next month's budget for a specific category
        
        Args:
            transaction_data (pd.DataFrame): Historical transaction data
            category (str): Category to predict
            confidence_interval (bool): Whether to calculate confidence interval
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained or category not in self.models:
            return {
                'predicted_amount': 0,
                'confidence': 'low',
                'error': f'No trained model for {category}'
            }
        
        # Prepare features
        X, y, dates = self.prepare_features(transaction_data, category)
        
        if len(X) == 0:
            # Fallback to simple average
            category_data = transaction_data[transaction_data['category'] == category]
            if len(category_data) > 0:
                avg_monthly = category_data['amount_abs'].sum() / max(1, len(category_data) / 30)
                return {
                    'predicted_amount': round(avg_monthly, 2),
                    'confidence': 'low',
                    'method': 'simple_average',
                    'fallback': True
                }
            else:
                return {
                    'predicted_amount': 100,
                    'confidence': 'very_low',
                    'method': 'default',
                    'error': 'No historical data'
                }
        
        # Use the most recent data point for prediction
        latest_features = X[-1].reshape(1, -1)
        latest_features_scaled = self.scalers[category].transform(latest_features)
        
        # Make prediction
        prediction = self.models[category].predict(latest_features_scaled)[0]
        prediction = max(prediction, 0)  # Ensure positive prediction
        
        # Calculate confidence based on historical accuracy
        if len(y) > 1:
            historical_predictions = self.models[category].predict(self.scalers[category].transform(X))
            mae = mean_absolute_error(y, historical_predictions)
            relative_error = mae / (y.mean() + 1e-6)
            
            if relative_error < 0.2:
                confidence = 'high'
            elif relative_error < 0.4:
                confidence = 'medium'
            else:
                confidence = 'low'
        else:
            confidence = 'medium'
        
        result = {
            'predicted_amount': round(prediction, 2),
            'confidence': confidence,
            'method': self.model_type,
            'historical_average': round(y.mean(), 2) if len(y) > 0 else 0,
            'training_months': len(y)
        }
        
        # Add confidence interval if requested
        if confidence_interval and len(y) > 2:
            std = np.std(y)
            result['confidence_interval'] = {
                'lower': round(max(0, prediction - 1.96 * std), 2),
                'upper': round(prediction + 1.96 * std, 2)
            }
        
        return result
    
    def predict_all_categories(self, transaction_data):
        """
        Predict next month's budget for all categories
        
        Args:
            transaction_data (pd.DataFrame): Historical transaction data
            
        Returns:
            dict: Predictions for all categories
        """
        predictions = {}
        
        for category in self.categories:
            predictions[category] = self.predict_next_month(transaction_data, category)
        
        # Calculate total predicted budget
        total_predicted = sum(
            pred['predicted_amount'] for pred in predictions.values()
            if pred['predicted_amount'] > 0
        )
        
        predictions['total_budget'] = {
            'predicted_amount': round(total_predicted, 2),
            'categories_count': len([p for p in predictions.values() if p['predicted_amount'] > 0])
        }
        
        return predictions
    
    def save_model(self, filepath='budget_predictor_model.pkl'):
        """Save the trained model to disk"""
        if not self.is_trained or not self.models:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'model_type': self.model_type,
            'categories': self.categories,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Budget predictor saved to {filepath}")
    
    def load_model(self, filepath='budget_predictor_model.pkl'):
        """Load a trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_type = model_data['model_type']
            self.categories = model_data['categories']
            self.is_trained = model_data['is_trained']
            
            print(f"Budget predictor loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"Model file {filepath} not found. Training new model...")
            self.train_model()
            self.save_model(filepath)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            self.train_model()
            self.save_model(filepath)

def main():
    """Test the budget predictor"""
    print("Testing Budget Predictor")
    print("=" * 50)
    
    # Test different model types
    model_types = ['linear_regression', 'random_forest']
    
    for model_type in model_types:
        print(f"\nTesting {model_type}...")
        
        # Create and train model
        predictor = BudgetPredictor(model_type=model_type)
        
        # Generate sample data
        sample_data = predictor.generate_sample_data(months=12)
        
        # Train model
        results = predictor.train_model(sample_data)
        
        # Test predictions
        predictions = predictor.predict_all_categories(sample_data)
        
        print(f"\nNext month predictions for {model_type}:")
        for category, pred in predictions.items():
            if category != 'total_budget' and pred['predicted_amount'] > 0:
                print(f"{category}: ${pred['predicted_amount']:.2f} (confidence: {pred['confidence']})")
        
        print(f"Total predicted budget: ${predictions['total_budget']['predicted_amount']:.2f}")
    
    print("\nBudget predictor testing complete!")

if __name__ == '__main__':
    main() 
