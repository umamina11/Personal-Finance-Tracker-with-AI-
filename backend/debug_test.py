import requests
import json

BASE_URL = 'http://localhost:5000/api'

def debug_test():
    print("🔍 Debugging ML Backend Issues")
    print("=" * 50)
    
    # Test 1: Basic health check
    try:
        response = requests.get('http://localhost:5000/health')
        print(f"✅ Health: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
        return
    
    # Test 2: ML Status
    try:
        response = requests.get(f'{BASE_URL}/ml/status')
        print(f"\n🤖 ML Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ ML Status failed: {e}")
    
    # Test 3: User Registration
    user_data = {
        'email': 'debug@example.com',
        'password': 'password123'
    }
    
    try:
        response = requests.post(f'{BASE_URL}/register', json=user_data)
        print(f"\n👤 Registration: {response.status_code}")
        if response.status_code == 201:
            result = response.json()
            token = result['access_token']
            print(f"✅ Got token: {token[:20]}...")
        elif response.status_code == 400:
            print("User might already exist, trying login...")
            response = requests.post(f'{BASE_URL}/login', json=user_data)
            print(f"Login: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                token = result['access_token']
                print(f"✅ Got token: {token[:20]}...")
            else:
                print(f"❌ Login failed: {response.text}")
                return
        else:
            print(f"❌ Registration failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Auth failed: {e}")
        return
    
    # Test 4: Simple transaction
    headers = {'Authorization': f'Bearer {token}'}
    transaction_data = {
        'amount': -10.50,
        'description': 'Test coffee purchase',
        'date': '2025-06-25'
    }
    
    try:
        response = requests.post(f'{BASE_URL}/transactions', json=transaction_data, headers=headers)
        print(f"\n💳 Transaction: {response.status_code}")
        if response.status_code == 201:
            result = response.json()
            print(f"✅ Transaction created: {result}")
        else:
            print(f"❌ Transaction failed: {response.text}")
            print(f"Request data: {transaction_data}")
            print(f"Headers: {headers}")
    except Exception as e:
        print(f"❌ Transaction error: {e}")
    
    # Test 5: Get transactions
    try:
        response = requests.get(f'{BASE_URL}/transactions', headers=headers)
        print(f"\n📋 Get Transactions: {response.status_code}")
        if response.status_code == 200:
            transactions = response.json()
            print(f"✅ Found {len(transactions)} transactions")
            if transactions:
                print(f"First transaction: {transactions[0]}")
        else:
            print(f"❌ Get transactions failed: {response.text}")
    except Exception as e:
        print(f"❌ Get transactions error: {e}")
    
    # Test 6: Insights
    try:
        response = requests.get(f'{BASE_URL}/insights', headers=headers)
        print(f"\n📊 Insights: {response.status_code}")
        if response.status_code == 200:
            insights = response.json()
            print(f"✅ Insights: {insights}")
        else:
            print(f"❌ Insights failed: {response.text}")
    except Exception as e:
        print(f"❌ Insights error: {e}")

if __name__ == '__main__':
    debug_test()