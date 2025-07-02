import os
import sqlite3
from datetime import datetime

def create_database_manually():
    """Manually create the database file and tables"""
    print("🔧 Manual Database Creation")
    print("=" * 40)
    
    # Get absolute path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(backend_dir, 'finance_tracker.db')
    
    print(f"📂 Backend directory: {backend_dir}")
    print(f"📂 Database path: {db_path}")
    
    # Remove existing file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🗑️  Removed existing database file")
    
    try:
        # Create database file manually
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("📝 Creating tables...")
        
        # Create User table
        cursor.execute('''
            CREATE TABLE user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(128) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ User table created")
        
        # Create Transaction table
        cursor.execute('''
            CREATE TABLE transaction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                account_id VARCHAR(100),
                amount FLOAT NOT NULL,
                description VARCHAR(200) NOT NULL,
                category VARCHAR(50),
                predicted_category VARCHAR(50),
                confidence FLOAT,
                date DATE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user (id)
            )
        ''')
        print("✅ Transaction table created")
        
        # Create Budget table
        cursor.execute('''
            CREATE TABLE budget (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category VARCHAR(50) NOT NULL,
                amount FLOAT NOT NULL,
                month INTEGER NOT NULL,
                year INTEGER NOT NULL,
                predicted BOOLEAN DEFAULT 0,
                confidence VARCHAR(20),
                FOREIGN KEY (user_id) REFERENCES user (id)
            )
        ''')
        print("✅ Budget table created")
        
        # Commit and close
        conn.commit()
        conn.close()
        
        # Verify file was created
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"🎉 SUCCESS! Database created at: {db_path}")
            print(f"📊 File size: {file_size} bytes")
            
            # Test the database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            
            print(f"📋 Tables in database: {[table[0] for table in tables]}")
            return True
            
        else:
            print("❌ Database file still not created!")
            return False
            
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

if __name__ == '__main__':
    success = create_database_manually()
    
    if success:
        print("\n✅ Database created successfully!")
        print("Now run: python app.py")
    else:
        print("\n❌ Database creation failed!")