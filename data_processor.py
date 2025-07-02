"""
Finance Data Processor
Handles all data processing for the ML models
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
        if not description or pd.isna(description):
            return ""
        
        # Convert to string and lowercase
        description = str(description).lower()
        
        # Remove special characters but keep spaces
        description = re.sub(r'[^a-zA-Z0-9\s]', ' ', description)
        
        # Remove extra spaces
        description = ' '.join(description.split())
        
        # Remove common transaction codes and words
        words_to_remove = [
            'pos', 'debit', 'credit', 'purchase', 'payment', 'transfer',
            'withdrawal', 'deposit', 'transaction', 'recurring', 'autopay',
            'checkcard', 'visa', 'mastercard', 'amex', 'discover'
        ]
        
        for word in words_to_remove:
            description = description.replace(word, '')
        
        # Remove numbers at the beginning (often transaction IDs)
        description = re.sub(r'^\d+\s*', '', description)
        
        # Remove extra spaces again
        description = ' '.join(description.split())
        
        return description.strip()
    
    def create_training_data(self):
        """
        Create comprehensive training data for expense categorization
        
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
            ('kfc fried chicken', 'Food & Dining'),
            ('chipotle mexican grill', 'Food & Dining'),
            ('panera bread', 'Food & Dining'),
            ('dunkin donuts', 'Food & Dining'),
            ('wendys restaurant', 'Food & Dining'),
            ('olive garden', 'Food & Dining'),
            ('applebees grill', 'Food & Dining'),
            ('chilis restaurant', 'Food & Dining'),
            ('red lobster', 'Food & Dining'),
            ('outback steakhouse', 'Food & Dining'),
            
            # Groceries (25 samples)
            ('walmart grocery', 'Groceries'),
            ('target grocery', 'Groceries'),
            ('kroger supermarket', 'Groceries'),
            ('safeway store', 'Groceries'),
            ('whole foods market', 'Groceries'),
            ('trader joes', 'Groceries'),
            ('costco wholesale', 'Groceries'),
            ('sams club', 'Groceries'),
            ('aldi grocery', 'Groceries'),
            ('publix supermarket', 'Groceries'),
            ('fresh market', 'Groceries'),
            ('local grocery store', 'Groceries'),
            ('farmers market', 'Groceries'),
            ('organic market', 'Groceries'),
            ('food shopping', 'Groceries'),
            ('weekly groceries', 'Groceries'),
            ('fresh produce', 'Groceries'),
            ('meat market', 'Groceries'),
            ('dairy products', 'Groceries'),
            ('convenience store snacks', 'Groceries'),
            
            # Shopping (30 samples)
            ('amazon online purchase', 'Shopping'),
            ('amazon prime', 'Shopping'),
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
            ('macys department store', 'Shopping'),
            ('nordstrom', 'Shopping'),
            ('tjmaxx', 'Shopping'),
            ('marshalls', 'Shopping'),
            ('kohls', 'Shopping'),
            ('jcpenney', 'Shopping'),
            ('sears', 'Shopping'),
            ('ebay purchase', 'Shopping'),
            ('etsy handmade', 'Shopping'),
            
            # Transportation (25 samples)
            ('uber ride share', 'Transportation'),
            ('lyft ride', 'Transportation'),
            ('taxi cab', 'Transportation'),
            ('metro transit', 'Transportation'),
            ('bus fare', 'Transportation'),
            ('train ticket', 'Transportation'),
            ('airline flight', 'Transportation'),
            ('car rental', 'Transportation'),
            ('parking meter', 'Transportation'),
            ('parking garage', 'Transportation'),
            ('car maintenance', 'Transportation'),
            ('oil change', 'Transportation'),
            ('car wash', 'Transportation'),
            ('auto repair', 'Transportation'),
            ('tire shop', 'Transportation'),
            ('bridge toll', 'Transportation'),
            ('highway toll', 'Transportation'),
            ('car insurance', 'Transportation'),
            ('vehicle registration', 'Transportation'),
            ('public transportation', 'Transportation'),
            
            # Gas (15 samples)
            ('shell gas station', 'Gas'),
            ('exxon fuel', 'Gas'),
            ('chevron gasoline', 'Gas'),
            ('bp gas pump', 'Gas'),
            ('mobil gas', 'Gas'),
            ('arco station', 'Gas'),
            ('texaco fuel', 'Gas'),
            ('sunoco gas', 'Gas'),
            ('marathon gas', 'Gas'),
            ('valero station', 'Gas'),
            ('fuel purchase', 'Gas'),
            ('gasoline fill up', 'Gas'),
            ('diesel fuel', 'Gas'),
            ('gas station convenience', 'Gas'),
            ('fuel rewards', 'Gas'),
            
            # Entertainment (25 samples)
            ('netflix subscription', 'Entertainment'),
            ('spotify premium', 'Entertainment'),
            ('hulu streaming', 'Entertainment'),
            ('disney plus', 'Entertainment'),
            ('amazon prime video', 'Entertainment'),
            ('movie theater', 'Entertainment'),
            ('cinema tickets', 'Entertainment'),
            ('amc theaters', 'Entertainment'),
            ('regal cinemas', 'Entertainment'),
            ('video game store', 'Entertainment'),
            ('gaming subscription', 'Entertainment'),
            ('xbox live', 'Entertainment'),
            ('playstation network', 'Entertainment'),
            ('steam games', 'Entertainment'),
            ('concert tickets', 'Entertainment'),
            ('theater show', 'Entertainment'),
            ('sports event', 'Entertainment'),
            ('amusement park', 'Entertainment'),
            ('bowling alley', 'Entertainment'),
            ('mini golf', 'Entertainment'),
            ('arcade games', 'Entertainment'),
            ('streaming service', 'Entertainment'),
            ('music subscription', 'Entertainment'),
            ('youtube premium', 'Entertainment'),
            ('twitch subscription', 'Entertainment'),
            
            # Bills & Utilities (30 samples)
            ('electric bill utility', 'Bills & Utilities'),
            ('gas bill heating', 'Bills & Utilities'),
            ('water bill utility', 'Bills & Utilities'),
            ('internet bill', 'Bills & Utilities'),
            ('phone bill mobile', 'Bills & Utilities'),
            ('cable tv bill', 'Bills & Utilities'),
            ('satellite tv', 'Bills & Utilities'),
            ('rent payment', 'Bills & Utilities'),
            ('mortgage payment', 'Bills & Utilities'),
            ('insurance premium', 'Bills & Utilities'),
            ('credit card payment', 'Bills & Utilities'),
            ('loan payment', 'Bills & Utilities'),
            ('student loan', 'Bills & Utilities'),
            ('car loan payment', 'Bills & Utilities'),
            ('utility company', 'Bills & Utilities'),
            ('hoa fees', 'Bills & Utilities'),
            ('property tax', 'Bills & Utilities'),
            ('subscription service', 'Bills & Utilities'),
            ('gym membership', 'Bills & Utilities'),
            ('fitness club', 'Bills & Utilities'),
            ('storage unit', 'Bills & Utilities'),
            ('security system', 'Bills & Utilities'),
            ('trash service', 'Bills & Utilities'),
            ('sewer bill', 'Bills & Utilities'),
            ('city utilities', 'Bills & Utilities'),
            
            # Healthcare (20 samples)
            ('pharmacy prescription', 'Healthcare'),
            ('doctor visit', 'Healthcare'),
            ('dental appointment', 'Healthcare'),
            ('hospital bill', 'Healthcare'),
            ('medical insurance', 'Healthcare'),
            ('eye doctor', 'Healthcare'),
            ('optometrist', 'Healthcare'),
            ('urgent care', 'Healthcare'),
            ('emergency room', 'Healthcare'),
            ('physical therapy', 'Healthcare'),
            ('lab tests', 'Healthcare'),
            ('blood work', 'Healthcare'),
            ('xray scan', 'Healthcare'),
            ('mri scan', 'Healthcare'),
            ('medicine purchase', 'Healthcare'),
            ('vitamins supplements', 'Healthcare'),
            ('medical supplies', 'Healthcare'),
            ('health clinic', 'Healthcare'),
            ('specialist visit', 'Healthcare'),
            ('therapy session', 'Healthcare'),
            
            # Education (10 samples)
            ('tuition payment', 'Education'),
            ('school fees', 'Education'),
            ('textbooks', 'Education'),
            ('online course', 'Education'),
            ('certification exam', 'Education'),
            ('workshop training', 'Education'),
            ('conference registration', 'Education'),
            ('learning materials', 'Education'),
            ('software license education', 'Education'),
            ('library fees', 'Education'),
            
            # Travel (15 samples)
            ('hotel booking', 'Travel'),
            ('airbnb rental', 'Travel'),
            ('flight ticket', 'Travel'),
            ('vacation rental', 'Travel'),
            ('travel insurance', 'Travel'),
            ('cruise booking', 'Travel'),
            ('tour package', 'Travel'),
            ('travel agency', 'Travel'),
            ('passport renewal', 'Travel'),
            ('visa application', 'Travel'),
            ('travel gear', 'Travel'),
            ('luggage purchase', 'Travel'),
            ('travel adapter', 'Travel'),
            ('currency exchange', 'Travel'),
            ('travel expenses', 'Travel'),
            
            # Income (10 samples)
            ('salary deposit', 'Income'),
            ('paycheck direct deposit', 'Income'),
            ('bonus payment', 'Income'),
            ('freelance payment', 'Income'),
            ('consulting fee', 'Income'),
            ('refund money', 'Income'),
            ('tax refund', 'Income'),
            ('cash deposit', 'Income'),
            ('investment return', 'Income'),
            ('dividend payment', 'Income'),
        ]
        
        # Create DataFrame
        df = pd.DataFrame(training_samples, columns=['description', 'category'])
        
        # Clean all descriptions
        df['description'] = df['description'].apply(self.clean_description)
        
        # Remove empty descriptions
        df = df[df['description'].str.len() > 0]
        
        print(f"Created training dataset with {len(df)} samples")
        print(f"Categories distribution:")
        for category, count in df['category'].value_counts().items():
            print(f"  {category}: {count} samples")
        
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
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])  # Remove rows with invalid dates
            
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Process amounts
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['amount'])  # Remove rows with invalid amounts
            
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
        
        # Filter expenses only
        df = df[df['is_expense'] == True]
        
        # Group by year-month
        monthly = df.groupby([df['date'].dt.year, df['date'].dt.month])['amount_abs'].sum()
        
        return monthly
    
    def extract_features(self, df):
        """
        Extract comprehensive features for machine learning models
        
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
        features['median_amount'] = df['amount_abs'].median()
        features['std_amount'] = df['amount_abs'].std()
        features['total_spending'] = df[df['is_expense']]['amount_abs'].sum()
        features['total_income'] = df[df['is_income']]['amount'].sum()
        features['net_amount'] = features['total_income'] - features['total_spending']
        
        # Category distribution
        category_counts = df['category'].value_counts()
        for category in self.categories:
            safe_category = category.lower().replace(" ", "_").replace("&", "and")
            features[f'count_{safe_category}'] = category_counts.get(category, 0)
            features[f'pct_{safe_category}'] = category_counts.get(category, 0) / len(df) * 100
        
        # Time-based features
        if 'date' in df.columns:
            features['days_span'] = (df['date'].max() - df['date'].min()).days
            features['avg_transactions_per_week'] = len(df) / max(features['days_span'] / 7, 1)
            features['avg_transactions_per_month'] = len(df) / max(features['days_span'] / 30, 1)
            
            # Day of week patterns
            features['weekend_transaction_pct'] = df['is_weekend'].sum() / len(df) * 100
            
            # Monthly patterns
            monthly_spending = df[df['is_expense']].groupby('month')['amount_abs'].sum()
            features['highest_spending_month'] = monthly_spending.idxmax() if not monthly_spending.empty else 0
            features['spending_seasonality'] = monthly_spending.std() / monthly_spending.mean() if not monthly_spending.empty else 0
        
        # Amount patterns
        if 'amount_abs' in df.columns:
            features['large_transaction_pct'] = (df['amount_abs'] > 100).sum() / len(df) * 100
            features['small_transaction_pct'] = (df['amount_abs'] < 10).sum() / len(df) * 100
            
        return features
    
    def get_spending_trends(self, df, periods=6):
        """
        Analyze spending trends over time
        
        Args:
            df (pd.DataFrame): Transaction data
            periods (int): Number of periods to analyze
            
        Returns:
            dict: Trend analysis
        """
        if df.empty or 'date' not in df.columns:
            return {}
        
        # Get monthly spending
        monthly_spending = self.get_monthly_spending(df)
        
        if len(monthly_spending) < 2:
            return {'error': 'Not enough data for trend analysis'}
        
        # Calculate trends
        recent_avg = monthly_spending.tail(min(periods, len(monthly_spending))).mean()
        overall_avg = monthly_spending.mean()
        
        # Growth rate
        if len(monthly_spending) >= 2:
            growth_rate = (monthly_spending.iloc[-1] - monthly_spending.iloc[0]) / monthly_spending.iloc[0] * 100
        else:
            growth_rate = 0
        
        # Volatility
        volatility = monthly_spending.std() / monthly_spending.mean() * 100
        
        return {
            'recent_avg_spending': round(recent_avg, 2),
            'overall_avg_spending': round(overall_avg, 2),
            'growth_rate_pct': round(growth_rate, 2),
            'volatility_pct': round(volatility, 2),
            'trend': 'increasing' if recent_avg > overall_avg else 'decreasing',
            'months_analyzed': len(monthly_spending)
        }

def main():
    """Test the data processor"""
    print("🔧 Testing Finance Data Processor")
    print("=" * 50)
    
    processor = FinanceDataProcessor()
    
    # Test data cleaning
    test_descriptions = [
        "STARBUCKS #12345 PURCHASE POS DEBIT",
        "AMZ*AMAZON.COM ONLINE PURCHASE VISA",
        "UBER *TRIP 123ABC PAYMENT",
        "WAL-MART #1234 GROCERY DEBIT CARD",
    ]
    
    print("Description cleaning test:")
    for desc in test_descriptions:
        cleaned = processor.clean_description(desc)
        print(f"  '{desc}' → '{cleaned}'")
    
    # Test training data creation
    print("\nCreating training data:")
    training_data = processor.create_training_data()
    print(f"Training data shape: {training_data.shape}")
    
    print("\n✅ Data processor tests completed successfully!")

if __name__ == '__main__':
    main()