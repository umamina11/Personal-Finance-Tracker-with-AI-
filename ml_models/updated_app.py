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

# Add ml_models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_models'))

# Import our advanced ML models
from model_trainer import FinanceMLTrainer
from data_processor import FinanceDataProcessor

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
    confidence = db.Column(db.Float, nullable=True)
    date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    year = db.Column(db.Integer, nullable=False)
    predicted = db.Column(db.Boolean, default=False)
    confidence = db.Column(db.String(20), nullable=True)

# Initialize ML Models
print("Initializing ML models...")
ml_trainer = FinanceMLTrainer(models_dir='models')
data_processor = FinanceDataProcessor()

# Load or train models
try:
    ml_trainer.load_models()
    print("✅ ML models loaded successfully")
except Exception as e:
    print(f"⚠️  Could not load models, training new ones: {e}")
    try:
        ml_trainer.train_all_models()
        print("✅ New ML models trained successfully")
    except Exception as e2:
        print(f"❌ Failed to train models: {e2}")
        ml_trainer = None

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
    user_id = int(get_jwt_identity())
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            description = data.get('description')
            amount = data.get('amount')
            user_category = data.get('category')
            
            # Use advanced ML prediction if no category provided
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
            
            transaction = Transaction(
                user_id=user_id,
                amount=amount,
                description=description,
                category=final_category,
                predicted_category=predicted_category,
                confidence=confidence,
                date=datetime.strptime(data.get('date'), '%Y-%m-%d').date()
            )
            
            db.session.add(transaction)
            db.session.commit()
            
            return jsonify({
                'message': 'Transaction created successfully',
                'transaction_id': transaction.id,
                'predicted_category': predicted_category,
                'confidence': confidence,
                'final_category': final_category
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
            'confidence': t.confidence,
            'date': t.date.isoformat(),
            'created_at': t.created_at.isoformat()
        } for t in transactions])

@app.route('/api/budget/predict', methods=['POST'])
@jwt_required()
def predict_budget():
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json()
        category = data.get('category')
        
        if not ml_trainer or not ml_trainer.budget_predictor:
            return jsonify({'error': 'Budget predictor not available'}), 500
        
        # Get user's transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if len(transactions) < 5:
            return jsonify({
                'predicted_amount': 100,
                'confidence': 'low',
                'message': 'Not enough historical data (need at least 5 transactions)',
                'method': 'default'
            })
        
        # Convert to DataFrame
        transaction_data = [{
            'date': t.date,
            'amount': abs(t.amount),
            'category': t.category or 'Other',
            'description': t.description
        } for t in transactions]
        
        df = data_processor.prepare_transaction_data(transaction_data)
        
        # Get prediction
        if category:
            prediction = ml_trainer.predict_budget(df, category)
        else:
            # Predict for all categories
            prediction = ml_trainer.predict_budget(df)
        
        return jsonify(prediction)
        
    except Exception as e:
        print(f"Budget prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/budget/predict-all', methods=['POST'])
@jwt_required()
def predict_all_budgets():
    try:
        user_id = int(get_jwt_identity())
        
        if not ml_trainer or not ml_trainer.budget_predictor:
            return jsonify({'error': 'Budget predictor not available'}), 500
        
        # Get user's transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if len(transactions) < 10:
            return jsonify({
                'error': 'Not enough historical data for comprehensive budget prediction',
                'required_transactions': 10,
                'current_transactions': len(transactions)
            })
        
        # Convert to DataFrame
        transaction_data = [{
            'date': t.date,
            'amount': abs(t.amount),
            'category': t.category or 'Other',
            'description': t.description
        } for t in transactions]
        
        df = data_processor.prepare_transaction_data(transaction_data)
        
        # Get predictions for all categories
        predictions = ml_trainer.predict_budget(df)
        
        return jsonify(predictions)
        
    except Exception as e:
        print(f"All budget prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/insights', methods=['GET'])
@jwt_required()
def get_insights():
    try:
        user_id = int(get_jwt_identity())
        
        # Get recent transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if not transactions:
            return jsonify({'message': 'No transactions found'})
        
        # Basic insights
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
                transaction_data = [{
                    'date': t.date,
                    'amount': abs(t.amount),
                    'category': t.category or 'Other',
                    'description': t.description
                } for t in transactions]
                
                df = data_processor.prepare_transaction_data(transaction_data)
                
                # Get budget predictions
                budget_predictions = ml_trainer.predict_budget(df)
                if 'error' not in budget_predictions:
                    advanced_insights['next_month_budget'] = budget_predictions
                
                # Additional ML insights could go here
                
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
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/ml/retrain', methods=['POST'])
@jwt_required()
def retrain_models():
    """Retrain ML models with user's actual data"""
    try:
        user_id = int(get_jwt_identity())
        
        # Get user's transactions
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if len(transactions) < 20:
            return jsonify({
                'error': 'Not enough data for retraining',
                'required_transactions': 20,
                'current_transactions': len(transactions)
            }), 400
        
        # Convert to training format
        training_data = []
        for t in transactions:
            if t.category and t.category != 'Other':
                training_data.append({
                    'description': t.description,
                    'category': t.category
                })
        
        if len(training_data) < 15:
            return jsonify({
                'error': 'Not enough categorized data for retraining',
                'required_categorized': 15,
                'current_categorized': len(training_data)
            }), 400
        
        # Retrain models
        df = pd.DataFrame(training_data)
        
        # Retrain expense categorizer
        ml_trainer.expense_categorizer.train_model(df)
        
        # Save updated models
        ml_trainer.expense_categorizer.save_model('models/expense_categorizer_retrained.pkl')
        
        return jsonify({
            'message': 'Models retrained successfully',
            'training_samples': len(training_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Get ML model status"""
    if ml_trainer:
        info = ml_trainer.get_model_info()
        return jsonify({
            'status': 'available',
            'models': info,
            'version': '2.0.0'
        })
    else:
        return jsonify({
            'status': 'unavailable',
            'error': 'ML models not loaded'
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'service': 'finance-tracker-api-ml',
        'ml_available': ml_trainer is not None,
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Personal Finance Tracker API with Advanced ML',
        'version': '2.0.0',
        'ml_features': [
            'AI-powered expense categorization',
            'Budget prediction',
            'Spending insights',
            'Model retraining'
        ],
        'endpoints': [
            '/api/register',
            '/api/login', 
            '/api/transactions',
            '/api/budget/predict',
            '/api/budget/predict-all',
            '/api/insights',
            '/api/ml/retrain',
            '/api/ml/status',
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
    
    print("🚀 Starting Personal Finance Tracker API with Advanced ML...")
    print("Available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("ML Status: http://localhost:5000/api/ml/status")
    
    # Run the app
    app.run(debug=True, port=5000, host='0.0.0.0')