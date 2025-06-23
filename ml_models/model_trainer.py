import os
import json
from datetime import datetime
import pandas as pd
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
            'model_path': model_path
        })
        
        print(f"✅ Expense categorizer trained and saved to {model_path}")
        return training_results
    
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
            'model_path': model_path
        })
        
        print(f"✅ Budget predictor trained and saved to {model_path}")
        return training_results
    
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
                'success': True,
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
                'success': True,
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
        self.expense_categorizer = ExpenseCategorizer(model_type=expense_model_type)
        expense_path = os.path.join(self.models_dir, f'expense_categorizer_{expense_model_type}.pkl')
        self.expense_categorizer.load_model(expense_path)
        
        # Load budget predictor
        self.budget_predictor = BudgetPredictor(model_type=budget_model_type)
        budget_path = os.path.join(self.models_dir, f'budget_predictor_{budget_model_type}.pkl')
        self.budget_predictor.load_model(budget_path)
        
        print("✅ Models loaded successfully")
    
    def predict_transaction_category(self, description):
        """
        Predict category for a transaction description
        
        Args:
            description (str): Transaction description
            
        Returns:
            dict: Prediction results
        """
        if self.expense_categorizer is None:
            return {'error': 'Expense categorizer not loaded'}
        
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
            return {'error': 'Budget predictor not loaded'}
        
        if category:
            return self.budget_predictor.predict_next_month(transaction_data, category)
        else:
            return self.budget_predictor.predict_all_categories(transaction_data)
    
    def evaluate_models(self, test_data=None):
        """
        Evaluate all trained models
        
        Args:
            test_data (dict, optional): Test data for evaluation
            
        Returns:
            dict: Evaluation results
        """
        results = {}
        
        # Evaluate expense categorizer
        if self.expense_categorizer and self.expense_categorizer.is_trained:
            if test_data and 'expense_data' in test_data:
                expense_eval = self.expense_categorizer.evaluate_on_data(test_data['expense_data'])
                results['expense_categorizer'] = expense_eval
            else:
                # Test on sample data
                test_descriptions = [
                    "Coffee at Starbucks",
                    "Amazon online purchase", 
                    "Uber ride",
                    "Grocery shopping",
                    "Electric bill"
                ]
                
                predictions = []
                for desc in test_descriptions:
                    pred = self.expense_categorizer.predict_category(desc)
                    predictions.append({
                        'description': desc,
                        'predicted_category': pred['category'],
                        'confidence': pred['confidence']
                    })
                
                results['expense_categorizer'] = {
                    'sample_predictions': predictions
                }
        
        # Evaluate budget predictor
        if self.budget_predictor and self.budget_predictor.is_trained:
            # Generate sample data for evaluation
            sample_data = self.budget_predictor.generate_sample_data(months=12)
            budget_predictions = self.budget_predictor.predict_all_categories(sample_data)
            
            results['budget_predictor'] = {
                'sample_predictions': budget_predictions
            }
        
        return results
    
    def save_training_summary(self):
        """Save training history and summary"""
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_directory': self.models_dir,
            'training_history': self.training_history,
            'available_models': {
                'expense_categorizer': self.expense_categorizer is not None,
                'budget_predictor': self.budget_predictor is not None
            }
        }
        
        summary_path = os.path.join(self.models_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📝 Training summary saved to {summary_path}")
    
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
            'models_directory': self.models_dir
        }
        
        return info

def main():
    """Demo and test the ML trainer"""
    print("🧪 Testing Finance ML Trainer")
    print("=" * 60)
    
    # Create trainer
    trainer = FinanceMLTrainer()
    
    # Train all models
    training_results = trainer.train_all_models()
    
    # Test predictions
    print("\n🔮 Testing Predictions")
    print("-" * 30)
    
    # Test expense categorization
    test_descriptions = [
        "Coffee at Starbucks downtown",
        "Amazon Prime monthly subscription", 
        "Uber ride to the airport",
        "Weekly grocery shopping at Safeway",
        "Monthly electric bill payment",
        "Movie tickets at AMC theater",
        "Gas fill-up at Shell station"
    ]
    
    print("\n💳 Expense Categorization Predictions:")
    for desc in test_descriptions:
        pred = trainer.predict_transaction_category(desc)
        if 'error' not in pred:
            print(f"'{desc}' → {pred['category']} (confidence: {pred['confidence']:.3f})")
        else:
            print(f"'{desc}' → Error: {pred['error']}")
    
    # Test budget prediction
    print("\n💰 Budget Predictions:")
    sample_data = trainer.budget_predictor.generate_sample_data(months=6)
    budget_preds = trainer.predict_budget(sample_data)
    
    if 'error' not in budget_preds:
        for category, pred in budget_preds.items():
            if category != 'total_budget' and pred.get('predicted_amount', 0) > 0:
                print(f"{category}: ${pred['predicted_amount']:.2f} (confidence: {pred['confidence']})")
        
        if 'total_budget' in budget_preds:
            print(f"Total Budget: ${budget_preds['total_budget']['predicted_amount']:.2f}")
    else:
        print(f"Budget prediction error: {budget_preds['error']}")
    
    # Model evaluation
    print("\n📊 Model Evaluation:")
    evaluation = trainer.evaluate_models()
    
    if 'expense_categorizer' in evaluation:
        print("Expense Categorizer: ✅ Working")
    
    if 'budget_predictor' in evaluation:
        print("Budget Predictor: ✅ Working")
    
    # Model info
    print("\n📋 Model Information:")
    info = trainer.get_model_info()
    print(json.dumps(info, indent=2))
    
    print("\n🎉 ML Trainer testing complete!")
    print(f"Models saved in: {trainer.models_dir}")

if __name__ == '__main__':
    main() 
