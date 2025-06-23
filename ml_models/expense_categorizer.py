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
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
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
                max_df=0.95  # Ignore terms that appear in more than 95% of documents
            )),
            ('classifier', self.classifier)
        ])
        
        return pipeline
    
    def train_model(self, training_data=None):
        """
        Train the expense categorization model
        
        Args:
            training_data (pd.DataFrame, optional): Custom training data
                                                  If None, uses built-in training data
        
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
        
        print(f"Training on {len(training_data)} samples")
        print(f"Categories: {training_data['category'].nunique()}")
        
        # Split data for evaluation
        X = training_data['description']
        y = training_data['category']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train pipeline
        self.model = self.create_pipeline()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation for more robust evaluation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'classification_report': report,
            'training_samples': len(training_data),
            'test_samples': len(X_test)
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
            raise ValueError("Model not trained. Call train_model() first.")
        
        if not description:
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
            raise ValueError("Model not trained. Call train_model() first.")
        
        results = []
        for desc in descriptions:
            result = self.predict_category(desc)
            results.append(result)
        
        return results
    
    def get_feature_importance(self, top_n=20):
        """
        Get the most important features (words) for classification
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            dict: Feature importance by category
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        try:
            # Get feature names from TF-IDF vectorizer
            feature_names = self.model.named_steps['tfidf'].get_feature_names_out()
            
            # Get feature importance based on model type
            if self.model_type == 'random_forest':
                importance = self.model.named_steps['classifier'].feature_importances_
                top_indices = importance.argsort()[-top_n:][::-1]
                
                return {
                    'top_features': [
                        {'feature': feature_names[i], 'importance': float(importance[i])}
                        for i in top_indices
                    ]
                }
            
            elif self.model_type == 'logistic_regression':
                # For logistic regression, we can get coefficients per class
                classes = self.model.named_steps['classifier'].classes_
                coefficients = self.model.named_steps['classifier'].coef_
                
                class_features = {}
                for i, class_name in enumerate(classes):
                    class_coef = coefficients[i]
                    top_indices = class_coef.argsort()[-top_n:][::-1]
                    
                    class_features[class_name] = [
                        {'feature': feature_names[j], 'coefficient': float(class_coef[j])}
                        for j in top_indices
                    ]
                
                return class_features
            
            else:
                return {'message': f'Feature importance not available for {self.model_type}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def save_model(self, filepath='expense_categorizer_model.pkl'):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'categories': self.categories,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='expense_categorizer_model.pkl'):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found. Training new model...")
            self.train_model()
            self.save_model(filepath)
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
            self.train_model()
            self.save_model(filepath)
    
    def evaluate_on_data(self, test_data):
        """
        Evaluate the model on custom test data
        
        Args:
            test_data (pd.DataFrame): Test data with 'description' and 'category' columns
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if test_data.empty:
            return {'error': 'No test data provided'}
        
        X_test = test_data['description']
        y_test = test_data['category']
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'test_samples': len(test_data)
        }

def main():
    """Test the expense categorizer"""
    print("Testing Expense Categorizer")
    print("=" * 50)
    
    # Test different model types
    model_types = ['naive_bayes', 'random_forest', 'logistic_regression']
    
    for model_type in model_types:
        print(f"\nTesting {model_type}...")
        
        # Create and train model
        categorizer = ExpenseCategorizer(model_type=model_type)
        results = categorizer.train_model()
        
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
        
        print(f"\nPredictions for {model_type}:")
        for desc in test_descriptions:
            result = categorizer.predict_category(desc)
            print(f"'{desc}' → {result['category']} (confidence: {result['confidence']:.3f})")
    
    print("\nExpense categorizer testing complete!")

if __name__ == '__main__':
    main() 
