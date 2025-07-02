"""
Expense Categorizer
Advanced ML model for categorizing financial transactions
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from data_processor import FinanceDataProcessor

class ExpenseCategorizer:
    """
    Advanced ML model for categorizing financial transactions.
    
    This class uses machine learning to automatically categorize expenses
    based on transaction descriptions. It supports multiple algorithms
    and provides confidence scores for predictions.
    """
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the expense categorizer
        
        Args:
            model_type (str): Type of ML model to use
                            Options: 'naive_bayes', 'random_forest', 'logistic_regression'
        """
        self.model_type = model_type
        self.model = None
        self.data_processor = FinanceDataProcessor()
        self.categories = self.data_processor.categories
        self.is_trained = False
        
        # Choose the ML algorithm
        if model_type == 'naive_bayes':
            self.classifier = MultinomialNB(alpha=1.0)
        elif model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_pipeline(self):
        """
        Create the ML pipeline with text processing and classification
        
        Returns:
            Pipeline: Scikit-learn pipeline
        """
        # Create pipeline with text vectorization and classification
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2),  # Include both single words and pairs
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                strip_accents='ascii',
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only include words with 2+ letters
            )),
            ('classifier', self.classifier)
        ])
        
        return pipeline
    
    def train_model(self, training_data=None, validation_split=0.2):
        """
        Train the expense categorization model
        
        Args:
            training_data (pd.DataFrame, optional): Custom training data
                                                  If None, uses built-in training data
            validation_split (float): Fraction of data to use for validation
        
        Returns:
            dict: Training results and metrics
        """
        print(f"Training {self.model_type} expense categorizer...")
        
        # Get training data
        if training_data is None:
            training_data = self.data_processor.create_training_data()
        
        # Validate training data
        if training_data.empty:
            raise ValueError("No training data available")
        
        # Ensure we have the required columns
        if 'description' not in training_data.columns or 'category' not in training_data.columns:
            raise ValueError("Training data must have 'description' and 'category' columns")
        
        # Clean the data
        training_data = training_data.dropna(subset=['description', 'category'])
        training_data = training_data[training_data['description'].str.len() > 0]
        
        print(f"Training on {len(training_data)} samples")
        print(f"Categories: {training_data['category'].nunique()}")
        
        # Split data for evaluation
        X = training_data['description']
        y = training_data['category']
        
        # Check if we have enough samples for each category
        category_counts = y.value_counts()
        min_samples = category_counts.min()
        
        if min_samples < 2:
            print("Warning: Some categories have very few samples")
            # Filter out categories with only 1 sample for better cross-validation
            valid_categories = category_counts[category_counts >= 2].index
            mask = y.isin(valid_categories)
            X = X[mask]
            y = y[mask]
            print(f"Filtered to {len(X)} samples with adequate category representation")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails, use random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
        
        # Create and train pipeline
        self.model = self.create_pipeline()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation for more robust evaluation
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(X)//10))
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            cv_scores = np.array([accuracy])
        
        # Classification report
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        except Exception as e:
            print(f"Classification report failed: {e}")
            report = {'accuracy': accuracy}
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'classification_report': report,
            'training_samples': len(training_data),
            'test_samples': len(X_test),
            'categories_count': y.nunique(),
            'model_type': self.model_type
        }
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results
    
    def predict_category(self, description):
        """
        Predict the category for a transaction description
        
        Args:
            description (str): Transaction description
            
        Returns:
            dict: Prediction results with category, confidence, and alternatives
        """
        if not self.is_trained or self.model is None:
            # Return a basic fallback prediction
            return {
                'category': 'Other',
                'confidence': 0.1,
                'alternatives': [],
                'error': 'Model not trained'
            }
        
        if not description or pd.isna(description):
            return {
                'category': 'Other',
                'confidence': 0.0,
                'alternatives': [],
                'error': 'Empty description'
            }
        
        # Clean the description
        cleaned_description = self.data_processor.clean_description(description)
        
        if not cleaned_description:
            return {
                'category': 'Other',
                'confidence': 0.5,
                'alternatives': [],
                'error': 'Description too short after cleaning'
            }
        
        try:
            # Get prediction
            prediction = self.model.predict([cleaned_description])[0]
            
            # Get confidence scores for all categories
            probabilities = self.model.predict_proba([cleaned_description])[0]
            
            # Get class labels (categories)
            classes = self.model.classes_
            
            # Create probability dictionary
            prob_dict = dict(zip(classes, probabilities))
            
            # Sort by confidence
            sorted_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Get top 3 alternatives
            alternatives = [
                {'category': cat, 'confidence': float(conf)} 
                for cat, conf in sorted_predictions[1:4]
            ]
            
            return {
                'category': prediction,
                'confidence': float(max(probabilities)),
                'alternatives': alternatives,
                'cleaned_description': cleaned_description,
                'all_probabilities': {k: float(v) for k, v in prob_dict.items()}
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'category': 'Other',
                'confidence': 0.0,
                'alternatives': [],
                'error': str(e)
            }
    
    def predict_batch(self, descriptions):
        """
        Predict categories for multiple descriptions at once
        
        Args:
            descriptions (list): List of transaction descriptions
            
        Returns:
            list: List of prediction results
        """
        if not self.is_trained or self.model is None:
            return [{'category': 'Other', 'confidence': 0.0, 'error': 'Model not trained'} 
                   for _ in descriptions]
        
        results = []
        for desc in descriptions:
            result = self.predict_category(desc)
            results.append(result)
        
        return results
    
    def save_model(self, filepath='expense_categorizer_model.pkl'):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'categories': self.categories,
            'is_trained': self.is_trained,
            'version': '1.0'
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath='expense_categorizer_model.pkl'):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found. Training new model...")
            try:
                self.train_model()
                self.save_model(filepath)
                return
            except Exception as e:
                print(f"Error training new model: {e}")
                return
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.categories = model_data['categories']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            try:
                self.train_model()
                self.save_model(filepath)
            except Exception as e2:
                print(f"Error training new model: {e2}")
    
    def retrain_with_user_data(self, user_transactions):
        """
        Retrain the model with user's corrected transaction data
        
        Args:
            user_transactions (pd.DataFrame): User transaction data with corrections
        """
        if user_transactions.empty:
            print("No user data provided for retraining")
            return False
        
        # Combine with existing training data
        base_training_data = self.data_processor.create_training_data()
        
        # Prepare user data
        user_training = user_transactions[['description', 'category']].copy()
        user_training['description'] = user_training['description'].apply(
            self.data_processor.clean_description
        )
        
        # Combine datasets
        combined_data = pd.concat([base_training_data, user_training], ignore_index=True)
        
        # Retrain model
        print(f"Retraining with {len(user_training)} user samples...")
        return self.train_model(combined_data)

def main():
    """Test the expense categorizer"""
    print("🤖 Testing Expense Categorizer")
    print("=" * 50)
    
    try:
        # Create and train model
        categorizer = ExpenseCategorizer(model_type='naive_bayes')
        results = categorizer.train_model()
        
        print(f"Training completed with accuracy: {results['accuracy']:.3f}")
        
        # Test predictions
        test_descriptions = [
            "Coffee at Starbucks downtown",
            "Amazon Prime subscription",
            "Uber ride to airport",
            "Grocery shopping at Whole Foods",
            "Electric bill payment",
            "Movie tickets at AMC",
            "Gas at Shell station"
        ]
        
        print(f"\nPredictions:")
        for desc in test_descriptions:
            result = categorizer.predict_category(desc)
            if 'error' not in result:
                print(f"  '{desc}' → {result['category']} (confidence: {result['confidence']:.3f})")
            else:
                print(f"  '{desc}' → Error: {result['error']}")
        
        print(f"✅ Expense categorizer test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ Expense categorizer testing complete!")

if __name__ == '__main__':
    main()