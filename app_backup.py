from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import os
from datetime import datetime, timedelta
import bcrypt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///finance_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string-change-this-in-production'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    transactions = db.relationship('Transaction', backref='user', lazy=True)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    account_id = db.Column(db.String(100), nullable=True)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=True)
    predicted_category = db.Column(db.String(50), nullable=True)
    date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    year = db.Column(db.Integer, nullable=False)

# Simple ML Model for Expense Categorization
class SimpleExpenseCategorizer:
    def __init__(self):
        self.model = None
        self.categories = [
            'Food & Dining', 'Shopping', 'Transportation', 'Entertainment',
            'Bills & Utilities', 'Healthcare', 'Education', 'Travel',
            'Groceries', 'Gas', 'Income', 'Other'
        ]
        self.training_data = [
            # Food & Dining
            ('starbucks', 'Food & Dining'), ('mcdonalds', 'Food & Dining'),
            ('pizza', 'Food & Dining'), ('restaurant', 'Food & Dining'),
            ('cafe', 'Food & Dining'), ('coffee', 'Food & Dining'),
            
            # Shopping
            ('amazon', 'Shopping'), ('target', 'Shopping'), ('walmart', 'Shopping'),
            ('best buy', 'Shopping'), ('clothing', 'Shopping'), ('store', 'Shopping'),
            
            # Transportation
            ('uber', 'Transportation'), ('gas station', 'Transportation'),
            ('metro', 'Transportation'), ('parking', 'Transportation'),
            ('car', 'Transportation'), ('fuel', 'Transportation'),
            
            # Entertainment
            ('netflix', 'Entertainment'), ('movie', 'Entertainment'),
            ('spotify', 'Entertainment'), ('game', 'Entertainment'),
            ('concert', 'Entertainment'), ('theater', 'Entertainment'),
            
            # Bills & Utilities
            ('electric', 'Bills & Utilities'), ('internet', 'Bills & Utilities'),
            ('phone bill', 'Bills & Utilities'), ('water', 'Bills & Utilities'),
            ('rent', 'Bills & Utilities'), ('utility', 'Bills & Utilities'),
            
            # Healthcare
            ('pharmacy', 'Healthcare'), ('doctor', 'Healthcare'),
            ('dental', 'Healthcare'), ('hospital', 'Healthcare'),
            ('medicine', 'Healthcare'), ('health', 'Healthcare'),
            
            # Groceries
            ('grocery', 'Groceries'), ('supermarket', 'Groceries'),
            ('whole foods', 'Groceries'), ('market', 'Groceries'),
            
            # Transportation/Gas
            ('shell', 'Gas'), ('exxon', 'Gas'), ('chevron', 'Gas'),
            
            # Income
            ('salary', 'Income'), ('paycheck', 'Income'), ('deposit', 'Income'),
        ]
    
    def train(self):
        """Train a simple keyword-based categorizer"""
        descriptions, categories = zip(*self.training_data)
        
        # Create TF-IDF pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        self.model.fit(descriptions, categories)
        print("Expense categorizer trained successfully!")
    
    def predict_category(self, description):
        """Predict category for a transaction description"""
        if self.model is None:
            self.train()
        
        try:
            prediction = self.model.predict([description.lower()])[0]
            probabilities = self.model.predict_proba([description.lower()])[0]
            confidence = max(probabilities)
            
            return {
                'category': prediction,
                'confidence': confidence
            }
        except:
            return {
                'category': 'Other',
                'confidence': 0.5
            }

# Initialize ML model
categorizer = SimpleExpenseCategorizer()

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = User(email=email, password_hash=password_hash)
        
        db.session.add(user)
        db.session.commit()
        
        access_token = create_access_token(identity=str(user.id))
        return jsonify({'access_token': access_token, 'user_id': user.id}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash):
            access_token = create_access_token(identity=str(user.id))
            return jsonify({'access_token': access_token, 'user_id': user.id}), 200
        
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/transactions', methods=['GET', 'POST'])
@jwt_required()
def transactions():
    user_id = int(get_jwt_identity())  # Convert string back to int
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            description = data.get('description')
            
            # Predict category using ML if not provided
            predicted_category = None
            if description and not data.get('category'):
                try:
                    prediction = categorizer.predict_category(description)
                    predicted_category = prediction['category']
                    print(f"Predicted category for '{description}': {predicted_category}")
                except Exception as e:
                    print(f"Error predicting category: {e}")
                    predicted_category = 'Other'
            
            transaction = Transaction(
                user_id=user_id,
                amount=data.get('amount'),
                description=description,
                category=data.get('category') or predicted_category,
                predicted_category=predicted_category,
                date=datetime.strptime(data.get('date'), '%Y-%m-%d').date()
            )
            
            db.session.add(transaction)
            db.session.commit()
            
            return jsonify({
                'message': 'Transaction created successfully',
                'predicted_category': predicted_category,
                'transaction_id': transaction.id
            }), 201
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    else:  # GET
        transactions = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.date.desc()).all()
        return jsonify([{
            'id': t.id,
            'amount': t.amount,
            'description': t.description,
            'category': t.category,
            'predicted_category': t.predicted_category,
            'date': t.date.isoformat(),
            'created_at': t.created_at.isoformat()
        } for t in transactions])

@app.route('/api/budget/predict', methods=['POST'])
@jwt_required()
def predict_budget():
    try:
        user_id = int(get_jwt_identity())  # Convert string back to int
        data = request.get_json()
        category = data.get('category')
        
        # Get user's transactions for this category
        transactions = Transaction.query.filter_by(
            user_id=user_id, 
            category=category
        ).order_by(Transaction.date.desc()).limit(100).all()
        
        if len(transactions) < 3:
            return jsonify({
                'predicted_amount': 100,
                'confidence': 'low',
                'message': 'Not enough historical data'
            })
        
        # Calculate simple average and trend
        amounts = [abs(t.amount) for t in transactions]
        average = sum(amounts) / len(amounts)
        
        # Simple prediction based on average
        prediction = average * 1.1  # 10% increase for next month
        
        return jsonify({
            'predicted_amount': round(prediction, 2),
            'confidence': 'high' if len(transactions) >= 10 else 'medium',
            'historical_average': round(average, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/insights', methods=['GET'])
@jwt_required()
def get_insights():
    try:
        user_id = int(get_jwt_identity())  # Convert string back to int
        
        # Get recent transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if not transactions:
            return jsonify({'message': 'No transactions found'})
        
        # Calculate insights
        total_spending = sum(abs(t.amount) for t in transactions if t.amount < 0)
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        
        # Category breakdown
        category_totals = {}
        for t in transactions:
            category = t.category or 'Other'
            if category not in category_totals:
                category_totals[category] = 0
            category_totals[category] += abs(t.amount)
        
        # Find top category
        top_category = max(category_totals.items(), key=lambda x: x[1])[0] if category_totals else None
        
        return jsonify({
            'total_spending': round(total_spending, 2),
            'total_income': round(total_income, 2),
            'net_amount': round(total_income - total_spending, 2),
            'category_breakdown': {k: round(v, 2) for k, v in category_totals.items()},
            'top_category': top_category,
            'transaction_count': len(transactions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'service': 'finance-tracker-api',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Personal Finance Tracker API',
        'version': '1.0.0',
        'endpoints': [
            '/api/register',
            '/api/login', 
            '/api/transactions',
            '/api/budget/predict',
            '/api/insights',
            '/health'
        ]
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize database
def init_db():
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Train ML model
    try:
        categorizer.train()
    except Exception as e:
        print(f"Warning: Could not train ML model: {e}")
    
    print("Starting Personal Finance Tracker API...")
    print("Available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    
    # Run the app
    app.run(debug=True, port=5000, host='0.0.0.0')