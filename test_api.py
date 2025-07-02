import requests
import json

BASE_URL = 'http://localhost:5000/api'

def test_health():
    """Test if the server is running"""
    try:
        response = requests.get('http://localhost:5000/health')
        print(f"✅ Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False

def test_registration():
    """Test user registration"""
    data = {
        'email': 'test@example.com',
        'password': 'password123'
    }
    try:
        response = requests.post(f'{BASE_URL}/register', json=data)
        print(f"\n✅ Registration: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        if 'access_token' in result:
            return result['access_token']
        return None
    except Exception as e:
        print(f"❌ Registration Failed: {e}")
        return None

def test_login():
    """Test user login"""
    data = {
        'email': 'test@example.com',
        'password': 'password123'
    }
    try:
        response = requests.post(f'{BASE_URL}/login', json=data)
        print(f"\n✅ Login: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        if 'access_token' in result:
            return result['access_token']
        return None
    except Exception as e:
        print(f"❌ Login Failed: {e}")
        return None

def test_add_transaction(token):
    """Test adding a transaction"""
    headers = {'Authorization': f'Bearer {token}'}
    data = {
        'amount': -25.50,
        'description': 'Coffee at Starbucks',
        'date': '2025-06-23'
    }
    try:
        response = requests.post(f'{BASE_URL}/transactions', json=data, headers=headers)
        print(f"\n✅ Add Transaction: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        return response.status_code == 201
    except Exception as e:
        print(f"❌ Add Transaction Failed: {e}")
        return False

def test_get_transactions(token):
    """Test getting transactions"""
    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(f'{BASE_URL}/transactions', headers=headers)
        print(f"\n✅ Get Transactions: {response.status_code}")
        result = response.json()
        print(f"Found {len(result)} transactions")
        if result:
            print(f"First transaction: {result[0]}")
        return True
    except Exception as e:
        print(f"❌ Get Transactions Failed: {e}")
        return False

def test_insights(token):
    """Test getting insights"""
    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(f'{BASE_URL}/insights', headers=headers)
        print(f"\n✅ Get Insights: {response.status_code}")
        result = response.json()
        print(f"Insights: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Get Insights Failed: {e}")
        return False

def test_budget_prediction(token):
    """Test budget prediction"""
    headers = {'Authorization': f'Bearer {token}'}
    data = {'category': 'Food & Dining'}
    try:
        response = requests.post(f'{BASE_URL}/budget/predict', json=data, headers=headers)
        print(f"\n✅ Budget Prediction: {response.status_code}")
        result = response.json()
        print(f"Prediction: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Budget Prediction Failed: {e}")
        return False

def main():
    print("🧪 Testing Personal Finance Tracker API")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Server is not running! Start the backend first.")
        return
    
    # Test 2: Registration
    token = test_registration()
    if not token:
        print("\n🔄 Trying login instead...")
        token = test_login()
    
    if not token:
        print("\n❌ Could not get authentication token!")
        return
    
    print(f"\n🔑 Authentication successful! Token: {token[:20]}...")
    
    # Test 3: Add some sample transactions
    sample_transactions = [
        {'amount': -25.50, 'description': 'Coffee at Starbucks', 'date': '2025-06-23'},
        {'amount': -120.00, 'description': 'Grocery shopping at Walmart', 'date': '2025-06-22'},
        {'amount': -15.75, 'description': 'Uber ride to work', 'date': '2025-06-21'},
        {'amount': -45.25, 'description': 'Amazon purchase', 'date': '2025-06-20'},
        {'amount': 2500.00, 'description': 'Salary deposit', 'date': '2025-06-15'},
    ]
    
    print(f"\n📝 Adding {len(sample_transactions)} sample transactions...")
    
    for i, transaction in enumerate(sample_transactions):
        headers = {'Authorization': f'Bearer {token}'}
        try:
            response = requests.post(f'{BASE_URL}/transactions', json=transaction, headers=headers)
            if response.status_code == 201:
                result = response.json()
                predicted = result.get('predicted_category', 'None')
                print(f"  ✅ Transaction {i+1}: {transaction['description']} → {predicted}")
            else:
                print(f"  ❌ Transaction {i+1} failed: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Transaction {i+1} error: {e}")
    
    # Test 4: Get all transactions
    test_get_transactions(token)
    
    # Test 5: Get insights
    test_insights(token)
    
    # Test 6: Budget prediction
    test_budget_prediction(token)
    
    print("\n🎉 API Testing Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. ✅ Backend is working!")
    print("2. 🌐 Set up the React frontend")
    print("3. 🔗 Connect frontend to backend")
    print("4. 🚀 Start using your finance tracker!")

if __name__ == '__main__':
    main()