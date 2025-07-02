#!/usr/bin/env python3
"""
Database creation script for Finance Tracker
This script creates the SQLite database with all required tables
"""

import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Create a minimal Flask app just for database creation
app = Flask(__name__)

# Database configuration
app.config['SECRET_KEY'] = 'temp-key-for-db-creation'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///finance_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define the database models (same as in main app)
class User(db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    transactions = db.relationship('Transaction', backref='user', lazy=True, cascade='all, delete-orphan')
    budgets = db.relationship('Budget', backref='user', lazy=True, cascade='all, delete-orphan')

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

def create_database():
    """Create the database and all tables"""
    print("🗄️  Creating Finance Tracker Database")
    print("=" * 50)
    
    # Print current directory and configuration
    print(f"Current directory: {os.getcwd()}")
    print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"Database file will be: {os.path.abspath('finance_tracker.db')}")
    
    try:
        with app.app_context():
            # Remove existing database if it exists
            if os.path.exists('finance_tracker.db'):
                print("🗑️  Removing existing database...")
                os.remove('finance_tracker.db')
            
            # Create all tables
            print("📊 Creating database tables...")
            db.create_all()
            
            # Verify tables were created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            print(f"✅ Created {len(tables)} tables:")
            for table in tables:
                columns = inspector.get_columns(table)
                print(f"  📋 {table}: {len(columns)} columns")
            
            print(f"\n🎉 Database created successfully!")
            
        # Verify file exists
        if os.path.exists('finance_tracker.db'):
            file_size = os.path.getsize('finance_tracker.db')
            print(f"✅ Database file created: finance_tracker.db ({file_size} bytes)")
            return True
        else:
            print("❌ Database file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to create database"""
    print("🚀 Finance Tracker Database Setup")
    print("=" * 50)
    
    # Create database
    db_created = create_database()
    
    if db_created:
        print("\n🎉 Database setup completed successfully!")
        print("You can now run your main application.")
        print("\nNext steps:")
        print("1. Create the ML module files")
        print("2. Run: python app.py")
        print("3. Test the API at: http://localhost:5000/health")
    else:
        print("\n❌ Database creation failed")
        print("Please check the errors above and try again")

if __name__ == '__main__':
    main()