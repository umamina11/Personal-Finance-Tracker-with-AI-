import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class FinanceDataProcessor:
    """
    Handles all data processing for our finance ML models.
    This class cleans, prepares, and formats data for machine learning.
    """
    
    def __init__(self):
        # Define categories we want to predict
        self.categories = [
            'Food & Dining', 'Shopping', 'Transportation', 'Entertainment',
            'Bills & Utilities', 'Healthcare', 'Education', 'Travel',
            'Groceries', 'Gas', 'Income', 'Other'
        ]
    
    def clean_description(self, description):
        """
        Clean transaction descriptions for better ML processing
        
        Args:
            description (str): Raw transaction description
            
        Returns:
            str: Cleaned description
        """
        if not description:
            return ""
        
        # Convert to lowercase
        description = description.lower()
        
        # Remove special characters but keep spaces
        description = re.sub(r'[^a-zA-Z0-9\s]', ' ', description)
        
        # Remove extra spaces
        description = ' '.join(description.split())
        
        # Remove common transaction codes
        words_to_remove = ['pos', 'debit', 'credit', 'purchase', 'payment', 'transfer']
        for word in words_to_remove:
            description = description.replace(word, '')
        
        # Remove extra spaces again
        description = ' '.join(description.split())
        
        return description.strip()
    
    def create_training_data(self):
        """
        Create sample training data for expense categorization.
        In a real app, this would come from user data or a larger dataset.
        
        Returns:
            pd.DataFrame: Training data with descriptions and categories
        """
        training_samples = [
            # Food & Dining (50 samples)
            ('starbucks coffee shop', 'Food & Dining'),
            ('mcdonalds restaurant', 'Food & Dining'),
            ('pizza hut delivery', 'Food & Dining'),
            ('subway sandwiches', 'Food & Dining'),
            ('local restaurant dinner', 'Food & Dining'),
            ('coffee shop downtown', 'Food & Dining'),
            ('burger king lunch', 'Food & Dining'),
            ('taco bell mexican food', 'Food & Dining'),
            ('chinese takeout', 'Food & Dining'),
            ('italian restaurant', 'Food & Dining'),
            ('sushi bar', 'Food & Dining'),
            ('food truck', 'Food & Dining'),
            ('cafe latte', 'Food & Dining'),
            ('bakery fresh bread', 'Food & Dining'),
            ('ice cream shop', 'Food & Dining'),
            ('diner breakfast', 'Food & Dining'),
            ('fast food drive thru', 'Food & Dining'),
            ('restaurant tip', 'Food & Dining'),
            ('lunch meeting', 'Food & Dining'),
            ('dinner date', 'Food & Dining'),
            
            # Shopping (30 samples)
            ('amazon online purchase', 'Shopping'),
            ('target store', 'Shopping'),
            ('walmart shopping', 'Shopping'),
            ('best buy electronics', 'Shopping'),
            ('clothing store', 'Shopping'),
            ('shoe store', 'Shopping'),
            ('department store', 'Shopping'),
            ('online shopping', 'Shopping'),
            ('retail store', 'Shopping'),
            ('mall shopping', 'Shopping'),
            ('electronics store', 'Shopping'),
            ('book store', 'Shopping'),
            ('jewelry store', 'Shopping'),
            ('home depot tools', 'Shopping'),
            ('lowes hardware', 'Shopping'),
            
            # Transportation (25 samples)
            ('uber ride share', 'Transportation'),
            ('lyft ride', 'Transportation'),
            ('taxi cab', 'Transportation'),
            ('gas station fuel', 'Transportation'),
            ('metro transit', 'Transportation'),
            ('bus fare', 'Transportation'),
            ('parking meter', 'Transportation'),
            ('parking garage', 'Transportation'),
            ('car maintenance', 'Transportation'),
            ('oil change', 'Transportation'),
            ('car wash', 'Transportation'),
            ('auto repair', 'Transportation'),
            ('tire shop', 'Transportation'),
            ('train ticket', 'Transportation'),
            ('airline flight', 'Transportation'),
            ('car rental', 'Transportation'),
            ('bridge toll', 'Transportation'),
            ('highway toll', 'Transportation'),
            ('car insurance', 'Transportation'),
            ('vehicle registration', 'Transportation'),
            
            # Entertainment (20 samples)
            ('netflix subscription', 'Entertainment'),
            ('spotify premium', 'Entertainment'),
            ('movie theater', 'Entertainment'),
            ('cinema tickets', 'Entertainment'),
            ('video game store', 'Entertainment'),
            ('gaming subscription', 'Entertainment'),
            ('concert tickets', 'Entertainment'),
            ('theater show', 'Entertainment'),
            ('sports event', 'Entertainment'),
            ('amusement park', 'Entertainment'),
            ('bowling alley', 'Entertainment'),
            ('mini golf', 'Entertainment'),
            ('arcade games', 'Entertainment'),
            ('streaming service', 'Entertainment'),
            ('music subscription', 'Entertainment'),
            
            # Bills & Utilities (20 samples)
            ('electric bill utility', 'Bills & Utilities'),
            ('gas bill heating', 'Bills & Utilities'),
            ('water bill utility', 'Bills & Utilities'),
            ('internet bill', 'Bills & Utilities'),
            ('phone bill mobile', 'Bills & Utilities'),
            ('cable tv bill', 'Bills & Utilities'),
            ('rent payment', 'Bills & Utilities'),
            ('mortgage payment', 'Bills & Utilities'),
            ('insurance premium', 'Bills & Utilities'),
            ('credit card payment', 'Bills & Utilities'),
            ('loan payment', 'Bills & Utilities'),
            ('utility company', 'Bills & Utilities'),
            ('hoa fees', 'Bills & Utilities'),
            ('property tax', 'Bills & Utilities'),
            ('subscription service', 'Bills & Utilities'),
            
            # Groceries (25 samples)
            ('grocery store', 'Groceries'),
            ('supermarket', 'Groceries'),
            ('whole foods market', 'Groceries'),
            ('safeway groceries', 'Groceries'),
            ('kroger supermarket', 'Groceries'),
            ('local market', 'Groceries'),
            ('farmers market', 'Groceries'),
            ('organic store', 'Groceries'),
            ('food shopping', 'Groceries'),
            ('weekly groceries', 'Groceries'),
            ('fresh produce', 'Groceries'),
            ('meat market', 'Groceries'),
            ('dairy products', 'Groceries'),
            ('bread bakery', 'Groceries'),
            ('convenience store', 'Groceries'),
            
            # Healthcare (15 samples)
            ('pharmacy prescription', 'Healthcare'),
            ('doctor visit', 'Healthcare'),
            ('dental appointment', 'Healthcare'),
            ('hospital bill', 'Healthcare'),
            ('medical insurance', 'Healthcare'),
            ('eye doctor', 'Healthcare'),
            ('urgent care', 'Healthcare'),
            ('physical therapy', 'Healthcare'),
            ('lab tests', 'Healthcare'),
            ('medicine purchase', 'Healthcare'),
            
            # Gas (10 samples)
            ('shell gas station', 'Gas'),
            ('exxon fuel', 'Gas'),
            ('chevron gasoline', 'Gas'),
            ('bp gas pump', 'Gas'),
            ('mobil gas', 'Gas'),
            ('arco station', 'Gas'),
            ('fuel purchase', 'Gas'),
            ('gasoline fill up', 'Gas'),
            
            # Income (10 samples)
            ('salary deposit', 'Income'),
            ('paycheck direct deposit', 'Income'),
            ('bonus payment', 'Income'),
            ('freelance payment', 'Income'),
            ('refund money', 'Income'),
            ('cash deposit', 'Income'),
            ('investment return', 'Income'),
            ('side job payment', 'Income'),
        ]
        
        # Create DataFrame
        df = pd.DataFrame(training_samples, columns=['description', 'category'])
        
        # Clean all descriptions
        df['description'] = df['description'].apply(self.clean_description)
        
        print(f"Created training dataset with {len(df)} samples")
        print(f"Categories: {df['category'].value_counts().to_dict()}")
        
        return df
    
    def prepare_transaction_data(self, transactions_list):
        """
        Prepare transaction data for ML processing
        
        Args:
            transactions_list (list): List of transaction dictionaries
            
        Returns:
            pd.DataFrame: Processed transaction data
        """
        if not transactions_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(transactions_list)
        
        # Clean descriptions
        if 'description' in df.columns:
            df['description_clean'] = df['description'].apply(self.clean_description)
        
        # Convert dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
        
        # Process amounts
        if 'amount' in df.columns:
            df['amount_abs'] = df['amount'].abs()
            df['is_expense'] = df['amount'] < 0
            df['is_income'] = df['amount'] > 0
        
        return df
    
    def get_monthly_spending(self, df, category=None):
        """
        Get monthly spending data for budget prediction
        
        Args:
            df (pd.DataFrame): Transaction data
            category (str, optional): Specific category to filter
            
        Returns:
            pd.Series: Monthly spending amounts
        """
        if df.empty:
            return pd.Series()
        
        # Filter by category if specified
        if category:
            df = df[df['category'] == category]
        
        # Group by year-month
        monthly = df.groupby([df['date'].dt.year, df['date'].dt.month])['amount_abs'].sum()
        
        return monthly
    
    def extract_features(self, df):
        """
        Extract features for machine learning models
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            dict: Extracted features
        """
        features = {}
        
        if df.empty:
            return features
        
        # Basic statistics
        features['total_transactions'] = len(df)
        features['avg_amount'] = df['amount_abs'].mean()
        features['total_spending'] = df[df['is_expense']]['amount_abs'].sum()
        features['total_income'] = df[df['is_income']]['amount'].sum()
        
        # Category distribution
        category_counts = df['category'].value_counts()
        for category in self.categories:
            features[f'count_{category.lower().replace(" ", "_").replace("&", "and")}'] = category_counts.get(category, 0)
        
        # Time-based features
        features['days_span'] = (df['date'].max() - df['date'].min()).days
        features['avg_transactions_per_week'] = len(df) / max(features['days_span'] / 7, 1)
        
        return features

if __name__ == '__main__':
    # Test the data processor
    processor = FinanceDataProcessor()
    
    # Test data cleaning
    test_descriptions = [
        "STARBUCKS #12345 PURCHASE",
        "AMZ*AMAZON.COM ONLINE",
        "UBER *TRIP 123ABC",
        "WAL-MART #1234 DEBIT"
    ]
    
    print("Testing description cleaning:")
    for desc in test_descriptions:
        cleaned = processor.clean_description(desc)
        print(f"'{desc}' → '{cleaned}'")
    
    # Test training data creation
    print("\nCreating training data:")
    training_data = processor.create_training_data()
    print(f"Sample data:")
    print(training_data.head())
    
    print("\nData processor tests completed successfully!") 
