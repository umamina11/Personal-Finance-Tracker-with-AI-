"""
Model Trainer
Unified trainer for all finance ML models
"""

import os
import json
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_processor import FinanceDataProcessor
from expense_categorizer import ExpenseCategorizer
from budget_predictor import BudgetPredictor

class FinanceMLTrainer:
    """
    Unified trainer for all finance ML models.
    
    This class orchestrates the training of both expense categorization
    and budget prediction models, manages model versions, and provides
    easy-to-use interfaces for the complete ML pipeline.
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the ML trainer
        
        Args:
            models_dir (str): Directory to save trained models
        """
        self.models_dir = models_dir
        self.data_processor = FinanceDataProcessor()
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.expense_categorizer = None
        self.budget_predictor = None
        
        # Training history
        self.training_history = []
    
    def train_expense_categorizer(self, model_type='naive_bayes', custom_data=None):
        """
        Train the expense categorization model
        
        Args:
            model_type (str): Type of model to train
            custom_data (pd.DataFrame, optional): Custom training data
            
        Returns:
            dict: Training results
        """
        print(f"\n🤖 Training Expense Categorizer ({model_type})")
        print("-" * 50)
        
        try:
            # Create model
            self.expense_categorizer = ExpenseCategorizer(model_type=model_type)
            
            # Train model
            training_results = self.expense_categorizer.train_model(custom_data)
            
            # Save model
            model_path = os.path.join(self.models_dir, f'expense_categorizer_{model_type}.pkl')
            self.expense_categorizer.save_model(model_path)
            
            # Update training history
            self.training_history.append({
                'model': 'expense_categorizer',
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'results': training_results,
                'model_path': model_path,
                'success': True
            })
            
            print(f"✅ Expense categorizer trained and saved to {model_path}")
            return training_results
            
        except Exception as e:
            error_msg = f"Error training expense categorizer: {e}"
            print(f"❌ {error_msg}")
            
            self.training_history.append({
                'model': 'expense_categorizer',
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            })
            
            # Return a basic result structure
            return {
                'error': error_msg,
                'accuracy': 0.0,
                'training_samples': 0
            }
    
    def train_budget_predictor(self, model_type='random_forest', custom_data=None):
        """
        Train the budget prediction model
        
        Args:
            model_type (str): Type of model to train
            custom_data (pd.DataFrame, optional): Custom training data
            
        Returns:
            dict: Training results
        """
        print(f"\n📊 Training Budget Predictor ({model_type})")
        print("-" * 50)
        
        try:
            # Create model
            self.budget_predictor = BudgetPredictor(model_type=model_type)
            
            # Train model
            training_results = self.budget_predictor.train_model(custom_data)
            
            # Save model
            model_path = os.path.join(self.models_dir, f'budget_predictor_{model_type}.pkl')
            self.budget_predictor.save_model(model_path)
            
            # Update training history
            self.training_history.append({
                'model': 'budget_predictor',
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'results': training_results,
                'model_path': model_path,
                'success': True
            })
            
            print(f"✅ Budget predictor trained and saved to {model_path}")
            return training_results
            
        except Exception as e:
            error_msg = f"Error training budget predictor: {e}"
            print(f"❌ {error_msg}")
            
            self.training_history.append({
                'model': 'budget_predictor',
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            })
            
            # Return a basic result structure
            return {
                'error': error_msg,
                'summary': {'trained_categories': 0}
            }
    
    def train_all_models(self, expense_model_type='naive_bayes', budget_model_type='random_forest'):
        """
        Train all ML models
        
        Args:
            expense_model_type (str): Model type for expense categorization
            budget_model_type (str): Model type for budget prediction
            
        Returns:
            dict: Complete training results
        """
        print("🚀 Training All Finance ML Models")
        print("=" * 60)
        
        results = {}
        
        # Train expense categorizer
        try:
            expense_results = self.train_expense_categorizer(expense_model_type)
            results['expense_categorizer'] = {
                'success': 'error' not in expense_results,
                'results': expense_results
            }
        except Exception as e:
            print(f"❌ Error training expense categorizer: {e}")
            results['expense_categorizer'] = {
                'success': False,
                'error': str(e)
            }
        
        # Train budget predictor
        try:
            budget_results = self.train_budget_predictor(budget_model_type)
            results['budget_predictor'] = {
                'success': 'error' not in budget_results,
                'results': budget_results
            }
        except Exception as e:
            print(f"❌ Error training budget predictor: {e}")
            results['budget_predictor'] = {
                'success': False,
                'error': str(e)
            }
        
        # Save training summary
        self.save_training_summary()
        
        print("\n🎉 Training Complete!")
        print(f"Models saved in: {self.models_dir}")
        
        return results
    
    def load_models(self, expense_model_type='naive_bayes', budget_model_type='random_forest'):
        """
        Load pre-trained models
        
        Args:
            expense_model_type (str): Model type for expense categorization
            budget_model_type (str): Model type for budget prediction
        """
        print("📥 Loading Pre-trained Models")
        print("-" * 30)
        
        # Load expense categorizer
        try:
            self.expense_categorizer = ExpenseCategorizer(model_type=expense_model_type)
            expense_path = os.path.join(self.models_dir, f'expense_categorizer_{expense_model_type}.pkl')
            self.expense_categorizer.load_model(expense_path)
            print(f"✅ Expense categorizer loaded")
        except Exception as e:
            print(f"⚠️  Could not load expense categorizer: {e}")
            print("Will train new model when needed")
        
        # Load budget predictor
        try:
            self.budget_predictor = BudgetPredictor(model_type=budget_model_type)
            budget_path = os.path.join(self.models_dir, f'budget_predictor_{budget_model_type}.pkl')
            self.budget_predictor.load_model(budget_path)
            print(f"✅ Budget predictor loaded")
        except Exception as e:
            print(f"⚠️  Could not load budget predictor: {e}")
            print("Will train new model when needed")
    
    def predict_transaction_category(self, description):
        """
        Predict category for a transaction description
        
        Args:
            description (str): Transaction description
            
        Returns:
            dict: Prediction results
        """
        if self.expense_categorizer is None:
            try:
                # Try to load model
                self.expense_categorizer = ExpenseCategorizer()
                self.expense_categorizer.load_model(
                    os.path.join(self.models_dir, 'expense_categorizer_naive_bayes.pkl')
                )
            except:
                return {'error': 'Expense categorizer not available', 'category': 'Other', 'confidence': 0.0}
        
        return self.expense_categorizer.predict_category(description)
    
    def predict_budget(self, transaction_data, category=None):
        """
        Predict budget for next month
        
        Args:
            transaction_data (pd.DataFrame): Historical transaction data
            category (str, optional): Specific category to predict
            
        Returns:
            dict: Budget predictions
        """
        if self.budget_predictor is None:
            try:
                # Try to load model
                self.budget_predictor = BudgetPredictor()
                self.budget_predictor.load_model(
                    os.path.join(self.models_dir, 'budget_predictor_random_forest.pkl')
                )
            except:
                return {'error': 'Budget predictor not available'}
        
        if category:
            return self.budget_predictor.predict_next_month(transaction_data, category)
        else:
            return self.budget_predictor.predict_all_categories(transaction_data)
    
    def save_training_summary(self):
        """Save training history and summary"""
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_directory': self.models_dir,
            'training_history': self.training_history,
            'available_models': {
                'expense_categorizer': {
                    'loaded': self.expense_categorizer is not None,
                    'trained': self.expense_categorizer.is_trained if self.expense_categorizer else False
                },
                'budget_predictor': {
                    'loaded': self.budget_predictor is not None,
                    'trained': self.budget_predictor.is_trained if self.budget_predictor else False
                }
            }
        }
        
        summary_path = os.path.join(self.models_dir, 'training_summary.json')
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📝 Training summary saved to {summary_path}")
        except Exception as e:
            print(f"⚠️  Could not save training summary: {e}")
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'expense_categorizer': {
                'loaded': self.expense_categorizer is not None,
                'trained': self.expense_categorizer.is_trained if self.expense_categorizer else False,
                'model_type': self.expense_categorizer.model_type if self.expense_categorizer else None
            },
            'budget_predictor': {
                'loaded': self.budget_predictor is not None,
                'trained': self.budget_predictor.is_trained if self.budget_predictor else False,
                'model_type': self.budget_predictor.model_type if self.budget_predictor else None
            },
            'models_directory': self.models_dir,
            'available_files': []
        }
        
        # List available model files
        try:
            if os.path.exists(self.models_dir):
                files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
                info['available_files'] = files
        except Exception as e:
            info['error'] = f"Could not list model files: {e}"
        
        return info
    
    def retrain_with_user_data(self, user_transactions):
        """
        Retrain models with user's actual transaction data
        
        Args:
            user_transactions (pd.DataFrame): User's transaction data
            
        Returns:
            dict: Retraining results
        """
        print("\n🔄 Retraining with User Data")
        print("-" * 30)
        
        if user_transactions.empty:
            return {'error': 'No user data provided'}
        
        results = {}
        
        # Retrain expense categorizer if we have categorized data
        categorized_data = user_transactions[
            user_transactions['category'].notna() & 
            (user_transactions['category'] != 'Other')
        ]
        
        if len(categorized_data) >= 10:
            try:
                if self.expense_categorizer is None:
                    self.expense_categorizer = ExpenseCategorizer()
                
                results['expense_categorizer'] = self.expense_categorizer.retrain_with_user_data(
                    categorized_data
                )
                print(f"✅ Expense categorizer retrained with {len(categorized_data)} samples")
            except Exception as e:
                results['expense_categorizer'] = {'error': str(e)}
                print(f"❌ Expense categorizer retraining failed: {e}")
        else:
            results['expense_categorizer'] = {
                'error': f'Need at least 10 categorized transactions, got {len(categorized_data)}'
            }
        
        # Retrain budget predictor if we have enough historical data
        if len(user_transactions) >= 30:
            try:
                if self.budget_predictor is None:
                    self.budget_predictor = BudgetPredictor()
                
                # Prepare data
                processed_data = self.data_processor.prepare_transaction_data(
                    user_transactions.to_dict('records')
                )
                
                results['budget_predictor'] = self.budget_predictor.train_model(processed_data)
                print(f"✅ Budget predictor retrained with {len(processed_data)} transactions")
            except Exception as e:
                results['budget_predictor'] = {'error': str(e)}
                print(f"❌ Budget predictor retraining failed: {e}")
        else:
            results['budget_predictor'] = {
                'error': f'Need at least 30 transactions, got {len(user_transactions)}'
            }
        
        return results
    
    def quick_setup(self):
        """
        Quick setup for development/testing - ensures models are available
        """
        print("⚡ Quick ML Setup")
        print("-" * 20)
        
        # Check if models exist, if not train them
        expense_model_path = os.path.join(self.models_dir, 'expense_categorizer_naive_bayes.pkl')
        budget_model_path = os.path.join(self.models_dir, 'budget_predictor_random_forest.pkl')
        
        models_exist = os.path.exists(expense_model_path) and os.path.exists(budget_model_path)
        
        if not models_exist:
            print("Training new models...")
            self.train_all_models()
        else:
            print("Loading existing models...")
            self.load_models()
        
        print("✅ ML setup complete!")

def main():
    """Demo and test the ML trainer"""
    print("🧪 Testing Finance ML Trainer")
    print("=" * 60)
    
    # Create trainer
    trainer = FinanceMLTrainer()
    
    # Quick setup
    trainer.quick_setup()
    
    # Test predictions
    print("\n🔮 Testing Predictions")
    print("-" * 30)
    
    # Test expense categorization
    test_descriptions = [
        "Coffee at Starbucks downtown location",
        "Amazon Prime monthly subscription renewal", 
        "Uber ride to the airport for business",
        "Weekly grocery shopping at Safeway",
        "Monthly electric utility bill payment",
        "Movie tickets at AMC theater",
        "Gas fill-up at Shell station"
    ]
    
    print("\n💳 Expense Categorization Predictions:")
    for desc in test_descriptions:
        pred = trainer.predict_transaction_category(desc)
        if 'error' not in pred:
            print(f"  '{desc}' → {pred['category']} (confidence: {pred['confidence']:.3f})")
        else:
            print(f"  '{desc}' → Error: {pred['error']}")
    
    # Test budget prediction
    print("\n💰 Budget Predictions:")
    try:
        if trainer.budget_predictor:
            sample_data = trainer.budget_predictor.generate_sample_data(months=6)
            budget_preds = trainer.predict_budget(sample_data)
            
            if 'error' not in budget_preds:
                categories_to_show = ['Food & Dining', 'Groceries', 'Transportation', 'Shopping', 'Bills & Utilities']
                for category in categories_to_show:
                    if category in budget_preds:
                        pred = budget_preds[category]
                        if pred.get('predicted_amount', 0) > 0:
                            print(f"  {category}: ${pred['predicted_amount']:.2f} (confidence: {pred.get('confidence', 'unknown')})")
                
                if 'total_budget' in budget_preds:
                    total = budget_preds['total_budget']
                    print(f"  Total Budget: ${total['predicted_amount']:.2f}")
            else:
                print(f"  Budget prediction error: {budget_preds['error']}")
        else:
            print("  Budget predictor not available")
    except Exception as e:
        print(f"  Budget prediction failed: {e}")
    
    # Model info
    print("\n📋 Model Information:")
    try:
        info = trainer.get_model_info()
        print(f"  Models directory: {info['models_directory']}")
        print(f"  Expense categorizer loaded: {info['expense_categorizer']['loaded']}")
        print(f"  Budget predictor loaded: {info['budget_predictor']['loaded']}")
        print(f"  Available model files: {len(info.get('available_files', []))}")
    except Exception as e:
        print(f"  Could not get model info: {e}")
    
    print("\n🎉 ML Trainer testing complete!")
    print(f"Models saved in: {trainer.models_dir}")

if __name__ == '__main__':
    main()