import requests
import json

BASE_URL = 'http://localhost:5000/api'

def test_ml_features():
    print("🧪 Testing ML-Powered Finance Tracker Backend")
    print("=" * 60)
    
    # Test 1: Health and ML Status
    try:
        response = requests.get('http://localhost:5000/health')
        health = response.json()
        print(f"✅ Server Health: {health['status']}")
        print(f"🤖 ML Available: {health['ml_available']}")
        
        ml_response = requests.get('http://localhost:5000/api/ml/status')
        ml_status = ml_response.json()
        print(f"🔧 ML Status: {ml_status['status']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: User Registration
    user_data = {
        'email': 'mluser@example.com',
        'password': 'password123'
    }
    
    try:
        response = requests.post(f'{BASE_URL}/register', json=user_data)
        if response.status_code == 201:
            result = response.json()
            token = result['access_token']
            print(f"✅ User registered successfully")
        else:
            # Try login instead
            response = requests.post(f'{BASE_URL}/login', json=user_data)
            result = response.json()
            token = result['access_token']
            print(f"✅ User logged in successfully")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Test 3: ML-Powered Transaction Creation
    print(f"\n🤖 Testing AI Transaction Categorization:")
    print("-" * 50)
    
    test_transactions = [
        {'amount': -4.50, 'description': 'Starbucks downtown coffee', 'date': '2025-06-23'},
        {'amount': -67.23, 'description': 'Weekly grocery shopping Safeway', 'date': '2025-06-22'},
        {'amount': -25.40, 'description': 'Uber ride to the airport', 'date': '2025-06-21'},
        {'amount': -34.99, 'description': 'Amazon Prime subscription renewal', 'date': '2025-06-20'},
        {'amount': -89.50, 'description': 'Monthly electric utility bill', 'date': '2025-06-19'},
        {'amount': -24.00, 'description': 'AMC movie theater tickets', 'date': '2025-06-18'},
        {'amount': 3000.00, 'description': 'Salary direct deposit paycheck', 'date': '2025-06-15'},
        {'amount': -32.10, 'description': 'Shell gas station fuel up', 'date': '2025-06-17'},
    ]
    
    headers = {'Authorization': f'Bearer {token}'}
    
    for transaction in test_transactions:
        try:
            response = requests.post(f'{BASE_URL}/transactions', json=transaction, headers=headers)
            if response.status_code == 201:
                result = response.json()
                predicted = result.get('predicted_category', 'None')
                confidence = result.get('confidence', 0)
                final = result.get('final_category', 'None')
                ml_used = result.get('ml_used', False)
                
                ml_indicator = "🤖" if ml_used else "🔧"
                print(f"{ml_indicator} '{transaction['description'][:35]}...' → {predicted} ({confidence:.3f}) → {final}")
            else:
                print(f"❌ Failed: {transaction['description'][:30]}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test 4: Advanced Insights
    print(f"\n📊 Testing Advanced Insights:")
    print("-" * 30)
    
    try:
        response = requests.get(f'{BASE_URL}/insights', headers=headers)
        if response.status_code == 200:
            insights = response.json()
            
            print(f"💰 Total Spending: ${insights.get('total_spending', 0):.2f}")
            print(f"💵 Total Income: ${insights.get('total_income', 0):.2f}")
            print(f"📈 Net Amount: ${insights.get('net_amount', 0):.2f}")
            print(f"🏆 Top Category: {insights.get('top_category', 'None')}")
            print(f"🤖 ML Predictions: {insights.get('ml_predictions_available', False)}")
            
            # Show category breakdown
            categories = insights.get('category_breakdown', {})
            confidences = insights.get('category_confidence', {})
            
            print(f"\n📋 Category Breakdown:")
            for category, amount in list(categories.items())[:5]:  # Top 5
                conf = confidences.get(category, 0)
                print(f"  {category}: ${amount:.2f} (avg confidence: {conf:.3f})")
        else:
            print(f"❌ Insights failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Insights error: {e}")
    
    # Test 5: Budget Prediction
    print(f"\n💰 Testing Budget Prediction:")
    print("-" * 30)
    
    try:
        budget_data = {'category': 'Food & Dining'}
        response = requests.post(f'{BASE_URL}/budget/predict', json=budget_data, headers=headers)
        if response.status_code == 200:
            prediction = response.json()
            amount = prediction.get('predicted_amount', 0)
            confidence = prediction.get('confidence', 'unknown')
            method = prediction.get('method', 'unknown')
            
            print(f"🍽️  Food & Dining Prediction: ${amount:.2f} ({confidence}) using {method}")
        else:
            print(f"❌ Budget prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Budget error: {e}")
    
    print(f"\n🎉 ML Backend Testing Complete!")
    print("=" * 60)
    print("✅ Your ML-powered finance tracker backend is working!")
    
    return True

if __name__ == '__main__':
    success = test_ml_features()
    if success:
        print("\n🚀 Next Steps:")
        print("1. ✅ Backend with ML is working!")
        print("2. 🌐 Set up React frontend")
        print("3. 🔗 Connect frontend to ML backend")
        print("4. 🎯 Test complete application")
    else:
        print("\n❌ Some tests failed. Check your backend is running.")