"""
Budget Predictor
Advanced ML model for predicting future budget needs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
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
            self.base_model_class = LinearRegression
        elif model_type == 'random_forest':
            self.base_model_class = lambda: RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def generate_sample_data(self, months=24, user_id=1):
        """
        Generate realistic sample transaction data for training
        
        Args:
            months (int): Number of months of data to generate
            user_id (int): User ID for the transactions
            
        Returns:
            pd.DataFrame: Sample transaction data
        """
        np.random.seed(42)  # For reproducible results
        
        # Define spending patterns by category
        category_patterns = {
            'Food & Dining': {'base': 300, 'variation': 100, 'trend': 1.02, 'transactions_per_month': 12},
            'Groceries': {'base': 400, 'variation': 80, 'trend': 1.01, 'transactions_per_month': 8},
            'Transportation': {'base': 200, 'variation': 60, 'trend': 1.03, 'transactions_per_month': 6},
            'Shopping': {'base': 250, 'variation': 150, 'trend': 1.01, 'transactions_per_month': 5},
            'Bills & Utilities': {'base': 350, 'variation': 50, 'trend': 1.02, 'transactions_per_month': 4},
            'Entertainment': {'base': 150, 'variation': 80, 'trend': 1.01, 'transactions_per_month': 4},
            'Healthcare': {'base': 200, 'variation': 100, 'trend': 1.02, 'transactions_per_month': 2},
            'Gas': {'base': 120, 'variation': 40, 'trend': 1.03, 'transactions_per_month': 4},
            'Other': {'base': 100, 'variation': 50, 'trend': 1.01, 'transactions_per_month': 3},
            'Income': {'base': 3000, 'variation': 200, 'trend': 1.03, 'transactions_per_month': 2}
        }
        
        transactions = []
        start_date = datetime.now() - timedelta(days=months * 30)
        
        for month in range(months):
            month_start = start_date + timedelta(days=month * 30)
            
            for category, pattern in category_patterns.items():
                # Calculate monthly spending with trend
                base_amount = pattern['base'] * (pattern['trend'] ** month)
                
                # Add seasonal variation
                month_of_year = (month_start.month - 1) % 12
                seasonal_multipliers = [0.8, 0.8, 0.9, 1.0, 1.0, 1.1, 1.1, 1.0, 0.9, 1.0, 1.2, 1.3]
                seasonal_factor = seasonal_multipliers[month_of_year]
                
                # Add random variation
                monthly_amount = base_amount * seasonal_factor * np.random.normal(1, 0.2)
                monthly_amount = max(monthly_amount, pattern['base'] * 0.3)  # Minimum spending
                
                # Generate individual transactions for the month
                num_transactions = max(1, np.random.poisson(pattern['transactions_per_month']))
                
                for i in range(num_transactions):
                    # Random day in the month
                    day_offset = np.random.randint(0, min(30, (datetime.now() - month_start).days + 1))
                    transaction_date = month_start + timedelta(days=day_offset)
                    
                    # Skip future dates
                    if transaction_date > datetime.now():
                        continue
                    
                    # Random transaction amount
                    if num_transactions == 1:
                        amount = monthly_amount
                    else:
                        amount = monthly_amount / num_transactions * np.random.uniform(0.5, 1.5)
                    
                    amount = max(amount, 1)  # Minimum $1 transaction
                    
                    # Make expenses negative, income positive
                    if category == 'Income':
                        amount = abs(amount)
                    else:
                        amount = -abs(amount)
                    
                    transactions.append({
                        'user_id': user_id,
                        'date': transaction_date,
                        'amount': amount,
                        'category': category,
                        'description': f'{category} transaction'
                    })
        
        df = pd.DataFrame(transactions)
        df = df.sort_values('date').reset_index(drop=True)
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
        # Filter by category and expenses only (except for Income)
        if category == 'Income':
            category_data = df[(df['category'] == category) & (df['is_income'] == True)].copy()
        else:
            category_data = df[(df['category'] == category) & (df['is_expense'] == True)].copy()
        
        if len(category_data) == 0:
            return np.array([]), np.array([]), []
        
        # Group by month-year
        category_data['year_month'] = category_data['date'].dt.to_period('M')
        
        monthly_data = category_data.groupby('year_month').agg({
            'amount_abs': ['sum', 'count', 'mean', 'std'],
            'day_of_week': 'mean',
            'day_of_month': 'mean',
            'is_weekend': 'mean'
        }).round(2)
        
        # Flatten column names
        monthly_data.columns = [
            'total_spent', 'transaction_count', 'avg_amount', 'std_amount',
            'avg_day_of_week', 'avg_day_of_month', 'weekend_ratio'
        ]
        
        # Fill NaN values
        monthly_data['std_amount'] = monthly_data['std_amount'].fillna(0)
        
        # Ensure we have enough data
        if len(monthly_data) < lookback_months + 1:
            return np.array([]), np.array([]), []
        
        # Sort by date
        monthly_data = monthly_data.sort_index()
        
        # Create features and targets
        X, y, dates = [], [], []
        
        for i in range(lookback_months, len(monthly_data)):
            # Features: previous months data
            feature_vector = []
            
            # Historical spending features
            for j in range(lookback_months):
                month_idx = i - lookback_months + j
                month_data = monthly_data.iloc[month_idx]
                
                feature_vector.extend([
                    month_data['total_spent'],
                    month_data['transaction_count'],
                    month_data['avg_amount'],
                    month_data['std_amount'],
                    month_data['avg_day_of_week'],
                    month_data['avg_day_of_month'],
                    month_data['weekend_ratio']
                ])
            
            # Add time-based features
            current_date = monthly_data.index[i]
            month = current_date.month
            
            # Seasonal features
            feature_vector.extend([
                np.sin(2 * np.pi * month / 12),  # Seasonal sine
                np.cos(2 * np.pi * month / 12),  # Seasonal cosine
                month,  # Month number
                int(month in [11, 12, 1])  # Holiday season indicator
            ])
            
            # Trend features
            if i >= 3:
                recent_trend = monthly_data.iloc[i-3:i]['total_spent'].mean()
                long_term_avg = monthly_data.iloc[:i]['total_spent'].mean()
                trend_ratio = recent_trend / (long_term_avg + 1e-6)
            else:
                trend_ratio = 1.0
            
            feature_vector.extend([
                monthly_data.iloc[:i]['total_spent'].mean(),  # Historical average
                trend_ratio,  # Trend indicator
                monthly_data.iloc[max(0, i-6):i]['total_spent'].std()  # Recent volatility
            ])
            
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
        trained_models = 0
        
        # Train separate model for each category
        for category in self.categories:
            try:
                print(f"Training model for {category}...")
                
                # Prepare features for this category
                X, y, dates = self.prepare_features(transaction_data, category)
                
                if len(X) == 0:
                    print(f"  No sufficient data for {category}")
                    continue
                
                if len(X) < 3:
                    print(f"  Not enough data for {category} (need at least 3 months)")
                    continue
                
                # Handle edge cases
                if len(np.unique(y)) < 2:
                    print(f"  No variation in data for {category}")
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Create and train model
                if self.model_type == 'linear_regression':
                    model = LinearRegression()
                else:  # random_forest
                    model = self.base_model_class()
                
                model.fit(X_scaled, y)
                
                # Evaluate model
                y_pred = model.predict(X_scaled)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                # Calculate R² score (handle edge cases)
                try:
                    r2 = r2_score(y, y_pred)
                except:
                    r2 = 0.0
                
                # Store model and scaler
                self.models[category] = model
                self.scalers[category] = scaler
                trained_models += 1
                
                results[category] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2_score': r2,
                    'training_samples': len(X),
                    'avg_prediction': float(y_pred.mean()),
                    'avg_actual': float(y.mean()),
                    'feature_count': X.shape[1]
                }
                
                print(f"  MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.3f}")
                
            except Exception as e:
                print(f"  Error training {category}: {e}")
                continue
        
        self.is_trained = trained_models > 0
        
        if self.is_trained:
            print(f"Budget predictor trained for {trained_models} categories")
        else:
            print("No models could be trained")
        
        results['summary'] = {
            'total_categories': len(self.categories),
            'trained_categories': trained_models,
            'success_rate': trained_models / len(self.categories) * 100
        }
        
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
            # Fallback to simple average
            return self._fallback_prediction(transaction_data, category)
        
        try:
            # Prepare features
            X, y, dates = self.prepare_features(transaction_data, category)
            
            if len(X) == 0:
                return self._fallback_prediction(transaction_data, category)
            
            # Use the most recent data point for prediction
            latest_features = X[-1].reshape(1, -1)
            latest_features_scaled = self.scalers[category].transform(latest_features)
            
            # Make prediction
            prediction = self.models[category].predict(latest_features_scaled)[0]
            prediction = max(prediction, 0)  # Ensure positive prediction
            
            # Calculate confidence based on historical accuracy
            confidence = self._calculate_confidence(y, X, category)
            
            result = {
                'predicted_amount': round(prediction, 2),
                'confidence': confidence,
                'method': self.model_type,
                'historical_average': round(y.mean(), 2) if len(y) > 0 else 0,
                'training_months': len(y),
                'category': category
            }
            
            # Add confidence interval if requested
            if confidence_interval and len(y) > 2:
                std = np.std(y)
                result['confidence_interval'] = {
                    'lower': round(max(0, prediction - 1.96 * std), 2),
                    'upper': round(prediction + 1.96 * std, 2)
                }
            
            return result
            
        except Exception as e:
            print(f"Prediction error for {category}: {e}")
            return self._fallback_prediction(transaction_data, category)
    
    def _fallback_prediction(self, transaction_data, category):
        """Fallback prediction method using simple statistics"""
        try:
            # Filter category data
            if category == 'Income':
                category_data = transaction_data[
                    (transaction_data['category'] == category) & 
                    (transaction_data['is_income'] == True)
                ]
            else:
                category_data = transaction_data[
                    (transaction_data['category'] == category) & 
                    (transaction_data['is_expense'] == True)
                ]
            
            if len(category_data) > 0:
                # Calculate monthly average
                monthly_spending = self.data_processor.get_monthly_spending(category_data)
                if len(monthly_spending) > 0:
                    avg_monthly = monthly_spending.mean()
                    recent_avg = monthly_spending.tail(3).mean() if len(monthly_spending) >= 3 else avg_monthly
                    
                    # Simple trend adjustment
                    prediction = recent_avg * 1.02  # Small growth assumption
                    
                    return {
                        'predicted_amount': round(prediction, 2),
                        'confidence': 'low',
                        'method': 'simple_average',
                        'historical_average': round(avg_monthly, 2),
                        'fallback': True,
                        'category': category
                    }
            
            # Default fallback
            default_amounts = {
                'Food & Dining': 250,
                'Groceries': 300,
                'Transportation': 150,
                'Shopping': 200,
                'Bills & Utilities': 300,
                'Entertainment': 100,
                'Healthcare': 150,
                'Gas': 100,
                'Other': 100,
                'Income': 2500
            }
            
            return {
                'predicted_amount': default_amounts.get(category, 100),
                'confidence': 'very_low',
                'method': 'default',
                'error': 'No historical data',
                'category': category
            }
            
        except Exception as e:
            return {
                'predicted_amount': 100,
                'confidence': 'very_low',
                'method': 'error_fallback',
                'error': str(e),
                'category': category
            }
    
    def _calculate_confidence(self, y_true, X, category):
        """Calculate confidence level based on model performance"""
        try:
            if len(y_true) < 2:
                return 'low'
            
            # Get model predictions for historical data
            X_scaled = self.scalers[category].transform(X)
            y_pred = self.models[category].predict(X_scaled)
            
            # Calculate relative error
            mae = mean_absolute_error(y_true, y_pred)
            relative_error = mae / (y_true.mean() + 1e-6)
            
            # Determine confidence level
            if relative_error < 0.15:
                return 'high'
            elif relative_error < 0.30:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'low'
    
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
        
        # Calculate total predicted budget (expenses only)
        total_predicted = sum(
            pred['predicted_amount'] for category, pred in predictions.items()
            if pred['predicted_amount'] > 0 and category != 'Income'
        )
        
        # Get income prediction
        income_prediction = predictions.get('Income', {}).get('predicted_amount', 0)
        
        predictions['total_budget'] = {
            'predicted_amount': round(total_predicted, 2),
            'predicted_income': round(income_prediction, 2),
            'predicted_savings': round(income_prediction - total_predicted, 2),
            'categories_count': len([p for p in predictions.values() if p.get('predicted_amount', 0) > 0])
        }
        
        return predictions
    
    def save_model(self, filepath='budget_predictor_model.pkl'):
        """Save the trained model to disk"""
        if not self.is_trained or not self.models:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'model_type': self.model_type,
            'categories': self.categories,
            'is_trained': self.is_trained,
            'version': '1.0'
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Budget predictor saved to {filepath}")
        except Exception as e:
            print(f"Error saving budget predictor: {e}")
            raise
    
    def load_model(self, filepath='budget_predictor_model.pkl'):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            print(f"Budget predictor file {filepath} not found. Training new model...")
            try:
                self.train_model()
                self.save_model(filepath)
                return
            except Exception as e:
                print(f"Error training new budget predictor: {e}")
                return
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_type = model_data['model_type']
            self.categories = model_data['categories']
            self.is_trained = model_data['is_trained']
            
            print(f"Budget predictor loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading budget predictor: {e}")
            print("Training new model...")
            try:
                self.train_model()
                self.save_model(filepath)
            except Exception as e2:
                print(f"Error training new budget predictor: {e2}")

def main():
    """Test the budget predictor"""
    print("📊 Testing Budget Predictor")
    print("=" * 50)
    
    try:
        # Create and train model
        predictor = BudgetPredictor(model_type='random_forest')
        
        # Generate sample data
        sample_data = predictor.generate_sample_data(months=12)
        
        # Train model
        results = predictor.train_model(sample_data)
        print(f"Training completed for {results['summary']['trained_categories']} categories")
        
        # Test predictions
        predictions = predictor.predict_all_categories(sample_data)
        
        print(f"\nNext month predictions:")
        for category, pred in predictions.items():
            if category != 'total_budget' and pred.get('predicted_amount', 0) > 0:
                print(f"  {category}: ${pred['predicted_amount']:.2f} (confidence: {pred.get('confidence', 'unknown')})")
        
        if 'total_budget' in predictions:
            total_info = predictions['total_budget']
            print(f"\nBudget Summary:")
            print(f"  Total expenses: ${total_info['predicted_amount']:.2f}")
            print(f"  Predicted income: ${total_info['predicted_income']:.2f}")
            print(f"  Predicted savings: ${total_info['predicted_savings']:.2f}")
        
        print(f"✅ Budget predictor test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ Budget predictor testing complete!")

if __name__ == '__main__':
    main()