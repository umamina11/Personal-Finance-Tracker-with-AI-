"""
Personal Finance Tracker API with Advanced AI/ML
Complete Flask application with expense categorization and budget prediction
"""

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import os
import sys
from datetime import datetime, timedelta
import bcrypt
import pandas as pd
import numpy as np
import traceback

# Import our ML models
try:
    from model_trainer import FinanceMLTrainer
    from data_processor import FinanceDataProcessor
    ML_AVAILABLE = True
    print("✅ ML modules imported successfully")
except ImportError as e:
    print(f"⚠️  ML modules not available: {e}")
    ML_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///finance_tracker.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-string-change-this-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Database Models
class User(db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    transactions = db.relationship('Transaction', backref='user', lazy=True, cascade='all, delete-orphan')
    budgets = db.relationship('Budget', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }

class Transaction(db.Model):
    __tablename__ = 'transaction'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    account_id = db.Column(db.String(100), nullable=True)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=True)
    predicted_category = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'account_id': self.account_id,
            'amount': self.amount,
            'description': self.description,
            'category': self.category,
            'predicted_category': self.predicted_category,
            'confidence': self.confidence,
            'date': self.date.isoformat() if self.date else None,
            'created_at': self.created_at.isoformat()
        }

class Budget(db.Model):
    __tablename__ = 'budget'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    year = db.Column(db.Integer, nullable=False)
    predicted = db.Column(db.Boolean, default=False)
    confidence = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'category': self.category,
            'amount': self.amount,
            'month': self.month,
            'year': self.year,
            'predicted': self.predicted,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }

# Initialize ML Models
ml_trainer = None
data_processor = None

def initialize_ml():
    """Initialize ML models"""
    global ml_trainer, data_processor
    
    if not ML_AVAILABLE:
        print("⚠️  ML not available - running without AI features")
        return False
    
    try:
        print("🤖 Initializing ML models...")
        ml_trainer = FinanceMLTrainer(models_dir='models')
        data_processor = FinanceDataProcessor()
        
        # Quick setup - load or train models
        ml_trainer.quick_setup()
        
        print("✅ ML models initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize ML models: {e}")
        traceback.print_exc()
        return False

# Routes

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Create user
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = User(email=email, password_hash=password_hash)
        
        db.session.add(user)
        db.session.commit()
        
        # Create access token
        access_token = create_access_token(identity=str(user.id))
        
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict(),
            'message': 'User registered successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        user = User.query.filter_by(email=email).first()
        
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password_hash):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Create access token
        access_token = create_access_token(identity=str(user.id))
        
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict(),
            'message': 'Login successful'
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/transactions', methods=['GET', 'POST'])
@jwt_required()
def transactions():
    """Get or create transactions"""
    user_id = int(get_jwt_identity())
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            description = data.get('description', '').strip()
            amount = data.get('amount')
            user_category = data.get('category', '').strip()
            date_str = data.get('date')
            
            # Validation
            if not description:
                return jsonify({'error': 'Description is required'}), 400
            
            if amount is None:
                return jsonify({'error': 'Amount is required'}), 400
            
            try:
                amount = float(amount)
            except (ValueError, TypeError):
                return jsonify({'error': 'Amount must be a number'}), 400
            
            if not date_str:
                return jsonify({'error': 'Date is required'}), 400
            
            try:
                transaction_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'error': 'Date must be in YYYY-MM-DD format'}), 400
            
            # Use ML prediction if no category provided
            predicted_category = None
            confidence = None
            
            if description and not user_category and ml_trainer:
                try:
                    prediction = ml_trainer.predict_transaction_category(description)
                    if 'error' not in prediction:
                        predicted_category = prediction['category']
                        confidence = prediction['confidence']
                        print(f"🤖 AI Prediction: '{description}' → {predicted_category} ({confidence:.3f})")
                    else:
                        print(f"⚠️  Prediction error: {prediction['error']}")
                        predicted_category = 'Other'
                        confidence = 0.5
                except Exception as e:
                    print(f"❌ ML prediction failed: {e}")
                    predicted_category = 'Other'
                    confidence = 0.5
            
            # Use user category if provided, otherwise use prediction
            final_category = user_category or predicted_category or 'Other'
            
            # Create transaction
            transaction = Transaction(
                user_id=user_id,
                amount=amount,
                description=description,
                category=final_category,
                predicted_category=predicted_category,
                confidence=confidence,
                date=transaction_date
            )
            
            db.session.add(transaction)
            db.session.commit()
            
            return jsonify({
                'transaction': transaction.to_dict(),
                'predicted_category': predicted_category,
                'confidence': confidence,
                'final_category': final_category,
                'message': 'Transaction created successfully'
            }), 201
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Failed to create transaction: {str(e)}'}), 500
    
    else:  # GET
        try:
            # Query parameters
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)
            category = request.args.get('category')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            
            # Build query
            query = Transaction.query.filter_by(user_id=user_id)
            
            if category:
                query = query.filter(Transaction.category == category)
            
            if start_date:
                try:
                    start = datetime.strptime(start_date, '%Y-%m-%d').date()
                    query = query.filter(Transaction.date >= start)
                except ValueError:
                    return jsonify({'error': 'start_date must be in YYYY-MM-DD format'}), 400
            
            if end_date:
                try:
                    end = datetime.strptime(end_date, '%Y-%m-%d').date()
                    query = query.filter(Transaction.date <= end)
                except ValueError:
                    return jsonify({'error': 'end_date must be in YYYY-MM-DD format'}), 400
            
            # Execute query
            transactions = query.order_by(Transaction.date.desc()).offset(offset).limit(limit).all()
            total_count = query.count()
            
            return jsonify({
                'transactions': [t.to_dict() for t in transactions],
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get transactions: {str(e)}'}), 500

@app.route('/api/budget/predict', methods=['POST'])
@jwt_required()
def predict_budget():
    """Predict budget for a specific category"""
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json() or {}
        category = data.get('category')
        
        if not ml_trainer or not ml_trainer.budget_predictor:
            return jsonify({'error': 'Budget predictor not available'}), 503
        
        # Get user's transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if len(transactions) < 5:
            return jsonify({
                'predicted_amount': 100,
                'confidence': 'low',
                'message': 'Not enough historical data (need at least 5 transactions)',
                'method': 'default'
            }), 200
        
        # Convert to DataFrame
        transaction_data = [t.to_dict() for t in transactions]
        df = data_processor.prepare_transaction_data(transaction_data)
        
        # Get prediction
        if category:
            prediction = ml_trainer.predict_budget(df, category)
        else:
            prediction = ml_trainer.predict_budget(df)
        
        return jsonify(prediction), 200
        
    except Exception as e:
        print(f"Budget prediction error: {e}")
        return jsonify({'error': f'Budget prediction failed: {str(e)}'}), 500

@app.route('/api/budget/predict-all', methods=['POST'])
@jwt_required()
def predict_all_budgets():
    """Predict budgets for all categories"""
    try:
        user_id = int(get_jwt_identity())
        
        if not ml_trainer or not ml_trainer.budget_predictor:
            return jsonify({'error': 'Budget predictor not available'}), 503
        
        # Get user's transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if len(transactions) < 10:
            return jsonify({
                'error': 'Not enough historical data for comprehensive budget prediction',
                'required_transactions': 10,
                'current_transactions': len(transactions)
            }), 400
        
        # Convert to DataFrame
        transaction_data = [t.to_dict() for t in transactions]
        df = data_processor.prepare_transaction_data(transaction_data)
        
        # Get predictions for all categories
        predictions = ml_trainer.predict_budget(df)
        
        return jsonify(predictions), 200
        
    except Exception as e:
        print(f"All budget prediction error: {e}")
        return jsonify({'error': f'Budget prediction failed: {str(e)}'}), 500

@app.route('/api/insights', methods=['GET'])
@jwt_required()
def get_insights():
    """Get financial insights for user"""
    try:
        user_id = int(get_jwt_identity())
        
        # Get recent transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if not transactions:
            return jsonify({
                'message': 'No transactions found',
                'total_spending': 0,
                'total_income': 0,
                'net_amount': 0,
                'category_breakdown': {},
                'transaction_count': 0
            }), 200
        
        # Basic calculations
        total_spending = sum(abs(t.amount) for t in transactions if t.amount < 0)
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        
        # Category breakdown
        category_totals = {}
        category_confidence = {}
        
        for t in transactions:
            category = t.category or 'Other'
            if category not in category_totals:
                category_totals[category] = 0
                category_confidence[category] = []
            
            category_totals[category] += abs(t.amount)
            if t.confidence:
                category_confidence[category].append(t.confidence)
        
        # Calculate average confidence per category
        avg_confidence = {}
        for category, confidences in category_confidence.items():
            if confidences:
                avg_confidence[category] = sum(confidences) / len(confidences)
            else:
                avg_confidence[category] = 0.0
        
        # Top category
        top_category = max(category_totals.items(), key=lambda x: x[1])[0] if category_totals else None
        
        # Advanced insights using ML
        advanced_insights = {}
        if ml_trainer and len(transactions) >= 10:
            try:
                # Convert to DataFrame for ML analysis
                transaction_data = [t.to_dict() for t in transactions]
                df = data_processor.prepare_transaction_data(transaction_data)
                
                # Get spending trends
                trends = data_processor.get_spending_trends(df)
                if trends:
                    advanced_insights['spending_trends'] = trends
                
                # Get budget predictions
                budget_predictions = ml_trainer.predict_budget(df)
                if 'error' not in budget_predictions:
                    advanced_insights['next_month_budget'] = budget_predictions
                
            except Exception as e:
                print(f"Advanced insights error: {e}")
                advanced_insights['error'] = str(e)
        
        return jsonify({
            'total_spending': round(total_spending, 2),
            'total_income': round(total_income, 2),
            'net_amount': round(total_income - total_spending, 2),
            'category_breakdown': {k: round(v, 2) for k, v in category_totals.items()},
            'category_confidence': {k: round(v, 3) for k, v in avg_confidence.items()},
            'top_category': top_category,
            'transaction_count': len(transactions),
            'ml_predictions_available': ml_trainer is not None,
            'advanced_insights': advanced_insights
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get insights: {str(e)}'}), 500

@app.route('/api/ml/retrain', methods=['POST'])
@jwt_required()
def retrain_models():
    """Retrain ML models with user's actual data"""
    try:
        user_id = int(get_jwt_identity())
        
        if not ml_trainer:
            return jsonify({'error': 'ML trainer not available'}), 503
        
        # Get user's transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if len(transactions) < 20:
            return jsonify({
                'error': 'Not enough data for retraining',
                'required_transactions': 20,
                'current_transactions': len(transactions)
            }), 400
        
        # Convert to DataFrame
        transaction_data = pd.DataFrame([t.to_dict() for t in transactions])
        
        # Retrain models
        results = ml_trainer.retrain_with_user_data(transaction_data)
        
        return jsonify({
            'message': 'Models retrained successfully',
            'results': results,
            'training_samples': len(transaction_data),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Get ML model status"""
    if ml_trainer:
        info = ml_trainer.get_model_info()
        return jsonify({
            'status': 'available',
            'models': info,
            'version': '3.0.0'
        }), 200
    else:
        return jsonify({
            'status': 'unavailable',
            'error': 'ML models not loaded',
            'version': '3.0.0'
        }), 503

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available expense categories"""
    categories = [
        'Food & Dining', 'Shopping', 'Transportation', 'Entertainment',
        'Bills & Utilities', 'Healthcare', 'Education', 'Travel',
        'Groceries', 'Gas', 'Income', 'Other'
    ]
    return jsonify({'categories': categories}), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'finance-tracker-api-v3',
        'ml_available': ml_trainer is not None,
        'database_connected': True,
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0.0'
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Personal Finance Tracker API with Advanced AI/ML',
        'version': '3.0.0',
        'features': [
            'User authentication (JWT)',
            'Transaction management',
            'AI-powered expense categorization',
            'ML budget prediction',
            'Financial insights and analytics',
            'Model retraining with user data'
        ],
        'endpoints': {
            'auth': ['/api/register', '/api/login'],
            'transactions': ['/api/transactions'],
            'ml': [
                '/api/budget/predict',
                '/api/budget/predict-all',
                '/api/insights',
                '/api/ml/retrain',
                '/api/ml/status'
            ],
            'utility': ['/api/categories', '/health']
        },
        'ml_status': 'available' if ml_trainer else 'unavailable'
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized'}), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({'error': 'Forbidden'}), 403

# Initialize database
def init_db():
    """Initialize the database"""
    try:
        with app.app_context():
            db.create_all()
            print("✅ Database initialized successfully!")
            return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Starting Personal Finance Tracker API v3.0")
    print("=" * 60)
    
    # Initialize database
    if not init_db():
        print("❌ Failed to initialize database. Exiting.")
        sys.exit(1)
    
    # Initialize ML
    ml_success = initialize_ml()
    if ml_success:
        print("✅ ML models initialized")
    else:
        print("⚠️  Running without ML features")
    
    print("\n🌐 Server Information:")
    print(f"   URL: http://localhost:5000")
    print(f"   Health check: http://localhost:5000/health")
    print(f"   ML Status: http://localhost:5000/api/ml/status")
    print(f"   API Documentation: http://localhost:5000/")
    
    # Run the app
    app.run(
        debug=True,
        port=5000,
        host='0.0.0.0'
    )