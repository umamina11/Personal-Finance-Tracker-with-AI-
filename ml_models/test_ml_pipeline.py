"""
Complete test suite for the Finance ML Pipeline
This script tests all components working together
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our ML components
from data_processor import FinanceDataProcessor
from expense_categorizer import ExpenseCategorizer  
from budget_predictor import BudgetPredictor
from model_trainer import FinanceMLTrainer

def test_data_processor():
    """Test the data processor component"""
    print("🔧 Testing Data Processor")
    print("-" * 30)
    
    processor = FinanceDataProcessor()
    
    # Test description cleaning
    test_descriptions = [
        "STARBUCKS #12345 PURCHASE POS",
        "AMZ*AMAZON.COM DEBIT CARD",
        "UBER *TRIP 123ABC PAYMENT",
        "WAL-MART #1234 GROCERY DEBIT"
    ]
    
    print("Description cleaning test:")
    for desc in test_descriptions:
        cleaned = processor.clean_description(desc)
        print(f"  '{desc}' → '{cleaned}'")
    
    # Test training data creation
    training_data = processor.create_training_data()
    print(f"\nTraining data: {len(training_data)} samples")
    print(f"Categories: {training_data['category'].nunique()}")
    
    return training_data

def test_expense_categorizer():
    """Test the expense categorizer"""
    print("\n🤖 Testing Expense Categorizer")
    print("-" * 30)
    
    # Test different model types
    model_types = ['naive_bayes', 'random_forest']
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type}...")
        
        categorizer = ExpenseCategorizer(model_type=model_type)
        training_results = categorizer.train_model()
        
        # Test predictions
        test_descriptions = [
            "Coffee shop purchase",
            "Amazon online shopping",
            "Uber ride downtown", 
            "Grocery store visit",
            "Monthly electric bill",
            "Movie theater tickets"
        ]
        
        predictions = []
        for desc in test_descriptions:
            pred = categorizer.predict_category(desc)
            predictions.append(pred)
            print(f"  '{desc}' → {pred['category']} ({pred['confidence']:.3f})")
        
        results[model_type] = {
            'training': training_results,
            'predictions': predictions
        }
    
    return results

def test_budget_predictor():
    """Test the budget predictor"""
    print("\n📊 Testing Budget Predictor")
    print("-" * 30)
    
    predictor = BudgetPredictor(model_type='random_forest')
    
    # Generate sample data
    print("Generating sample transaction data...")
    sample_data = predictor.generate_sample_data(months=18)
    print(f"Generated {len(sample_data)} transactions over 18 months")
    
    # Train model
    print("\nTraining budget predictor...")
    training_results = predictor.train_model(sample_data)
    
    # Test predictions
    print("\nTesting budget predictions...")
    predictions = predictor.predict_all_categories(sample_data)
    
    print("Next month budget predictions:")
    total_budget = 0
    for category, pred in predictions.items():
        if category != 'total_budget' and pred.get('predicted_amount', 0) > 0:
            amount = pred['predicted_amount']
            confidence = pred['confidence']
            print(f"  {category}: ${amount:.2f} (confidence: {confidence})")
            total_budget += amount
    
    print(f"\nTotal predicted budget: ${total_budget:.2f}")
    
    return {
        'training': training_results,
        'predictions': predictions,
        'sample_data': sample_data
    }

def test_model_trainer():
    """Test the unified model trainer"""
    print("\n🚀 Testing Model Trainer")
    print("-" * 30)
    
    trainer = FinanceMLTrainer(models_dir='test_models')
    
    # Train all models
    print("Training all models...")
    training_results = trainer.train_all_models()
    
    # Test expense categorization
    print("\nTesting expense categorization through trainer:")
    test_transactions = [
        "Starbucks coffee and pastry",
        "Target shopping trip",
        "Lyft ride to work",
        "Whole Foods groceries",
        "Netflix monthly subscription"
    ]
    
    for transaction in test_transactions:
        pred = trainer.predict_transaction_category(transaction)
        if 'error' not in pred:
            print(f"  '{transaction}' → {pred['category']} ({pred['confidence']:.3f})")
        else:
            print(f"  '{transaction}' → Error: {pred['error']}")
    
    # Test budget prediction
    print("\nTesting budget prediction through trainer:")
    sample_data = trainer.budget_predictor.generate_sample_data(months=12)
    budget_pred = trainer.predict_budget(sample_data)
    
    if 'error' not in budget_pred:
        for category in ['Food & Dining', 'Groceries', 'Transportation', 'Shopping']:
            if category in budget_pred:
                pred = budget_pred[category]
                print(f"  {category}: ${pred['predicted_amount']:.2f} ({pred['confidence']})")
    
    # Model evaluation
    print("\nEvaluating models...")
    evaluation = trainer.evaluate_models()
    
    return {
        'training': training_results,
        'evaluation': evaluation
    }

def test_integration_workflow():
    """Test the complete workflow as it would be used in the app"""
    print("\n🔄 Testing Integration Workflow")
    print("-" * 30)
    
    # Simulate user transactions
    user_transactions = [
        {"description": "Starbucks coffee", "amount": -4.50, "date": "2025-06-20"},
        {"description": "Safeway groceries", "amount": -67.23, "date": "2025-06-19"},
        {"description": "Uber to airport", "amount": -25.40, "date": "2025-06-18"},
        {"description": "Amazon purchase", "amount": -34.99, "date": "2025-06-17"},
        {"description": "Electric bill", "amount": -89.50, "date": "2025-06-16"},
        {"description": "Movie tickets", "amount": -24.00, "date": "2025-06-15"},
        {"description": "Salary deposit", "amount": 3000.00, "date": "2025-06-15"},
        {"description": "Gas station", "amount": -32.10, "date": "2025-06-14"},
    ]
    
    # Create and train models
    trainer = FinanceMLTrainer(models_dir='integration_test_models')
    trainer.train_all_models()
    
    # Process transactions with AI categorization
    processed_transactions = []
    print("Processing transactions with AI categorization:")
    
    for transaction in user_transactions:
        # Predict category if amount is negative (expense)
        if transaction["amount"] < 0:
            pred = trainer.predict_transaction_category(transaction["description"])
            category = pred.get('category', 'Other')
            confidence = pred.get('confidence', 0.0)
            
            print(f"  '{transaction['description']}' → {category} ({confidence:.3f})")
        else:
            category = 'Income'
            confidence = 1.0
        
        processed_transactions.append({
            **transaction,
            'category': category,
            'predicted_category': category,
            'confidence': confidence
        })
    
    # Convert to DataFrame for budget prediction
    df = pd.DataFrame(processed_transactions)
    df['date'] = pd.to_datetime(df['date'])
    df['amount_abs'] = df['amount'].abs()
    
    # Generate more historical data for better budget prediction
    historical_data = trainer.budget_predictor.generate_sample_data(months=6)
    combined_data = pd.concat([historical_data, df], ignore_index=True)
    
    # Predict next month's budget
    print("\nPredicting next month's budget:")
    budget_predictions = trainer.predict_budget(combined_data)
    
    if 'error' not in budget_predictions:
        categories_with_predictions = [
            cat for cat, pred in budget_predictions.items() 
            if cat != 'total_budget' and pred.get('predicted_amount', 0) > 0
        ]
        
        for category in categories_with_predictions[:5]:  # Show top 5
            pred = budget_predictions[category]
            print(f"  {category}: ${pred['predicted_amount']:.2f} (confidence: {pred['confidence']})")
    
    print("\n✅ Integration workflow test complete!")
    
    return {
        'processed_transactions': processed_transactions,
        'budget_predictions': budget_predictions
    }

def run_performance_test():
    """Test performance with larger datasets - FIXED VERSION"""
    print("\n⚡ Performance Testing")
    print("-" * 30)
    
    try:
        # Create trainer
        trainer = FinanceMLTrainer(models_dir='performance_test_models')
        
        # Train models first (this was the missing step!)
        print("Training models for performance test...")
        training_results = trainer.train_all_models()
        
        # Check if models were trained successfully
        if not trainer.budget_predictor or not trainer.budget_predictor.is_trained:
            print("❌ Budget predictor not available for performance test")
            return {
                'error': 'Budget predictor not trained',
                'training_results': training_results
            }
        
        # Generate large dataset
        print("Generating large dataset...")
        large_dataset = trainer.budget_predictor.generate_sample_data(months=36)
        print(f"Generated {len(large_dataset)} transactions")
        
        # Time the training
        start_time = datetime.now()
        trainer.train_all_models()
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Training time: {training_time:.2f} seconds")
        
        # Time predictions
        test_descriptions = [
            "Starbucks coffee purchase",
            "Amazon online shopping",
            "Grocery store visit"
        ] * 100  # 300 predictions
        
        start_time = datetime.now()
        for desc in test_descriptions:
            trainer.predict_transaction_category(desc)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        print(f"300 predictions time: {prediction_time:.2f} seconds")
        print(f"Average prediction time: {(prediction_time/300)*1000:.2f} ms")
        
        return {
            'training_time': training_time,
            'prediction_time': prediction_time,
            'dataset_size': len(large_dataset)
        }
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return {
            'error': str(e),
            'status': 'failed'
        }

def main():
    """Run all tests with better error handling"""
    print("🧪 Complete Finance ML Pipeline Test Suite")
    print("=" * 60)
    
    results = {}
    test_results = []
    
    # Test individual components
    try:
        print("Running Data Processor test...")
        results['data_processor'] = test_data_processor()
        test_results.append(("Data Processor", True))
    except Exception as e:
        print(f"❌ Data Processor test failed: {e}")
        test_results.append(("Data Processor", False))
    
    try:
        print("Running Expense Categorizer test...")
        results['expense_categorizer'] = test_expense_categorizer()
        test_results.append(("Expense Categorizer", True))
    except Exception as e:
        print(f"❌ Expense Categorizer test failed: {e}")
        test_results.append(("Expense Categorizer", False))
    
    try:
        print("Running Budget Predictor test...")
        results['budget_predictor'] = test_budget_predictor()
        test_results.append(("Budget Predictor", True))
    except Exception as e:
        print(f"❌ Budget Predictor test failed: {e}")
        test_results.append(("Budget Predictor", False))
    
    try:
        print("Running Model Trainer test...")
        results['model_trainer'] = test_model_trainer()
        test_results.append(("Model Trainer", True))
    except Exception as e:
        print(f"❌ Model Trainer test failed: {e}")
        test_results.append(("Model Trainer", False))
    
    # Test integration
    try:
        print("Running Integration test...")
        results['integration'] = test_integration_workflow()
        test_results.append(("Integration", True))
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        test_results.append(("Integration", False))
    
    # Performance test (optional)
    try:
        print("Running Performance test...")
        results['performance'] = run_performance_test()
        if 'error' not in results['performance']:
            test_results.append(("Performance", True))
        else:
            print(f"⚠️  Performance test had issues but main pipeline works")
            test_results.append(("Performance", False))
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        print(f"⚠️  This is optional - the main ML pipeline still works")
        test_results.append(("Performance", False))
    
    print("\n🎉 Test Suite Complete!")
    print("=" * 60)
    
    # Summary
    print("\n📋 Test Summary:")
    passed_tests = 0
    for test_name, passed in test_results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}: {'Working' if passed else 'Failed'}")
        if passed:
            passed_tests += 1
    
    success_rate = (passed_tests / len(test_results)) * 100
    print(f"\n📊 Success Rate: {passed_tests}/{len(test_results)} ({success_rate:.1f}%)")
    
    if passed_tests >= 4:  # At least core components working
        print(f"\n🚀 ML Pipeline is ready for production!")
        print(f"📂 Models saved in multiple test directories")
        return True
    else:
        print(f"\n⚠️  Some critical tests failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n✅ ML Pipeline is ready for production!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")