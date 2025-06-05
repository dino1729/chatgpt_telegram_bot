#!/usr/bin/env python3
"""
Test script to verify Bing API key is working
"""
import sys
import os
import requests
import yaml

def test_bing_api():
    """Test if Bing API key is working"""
    try:
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        bing_api_key = config.get('bing_api_key', '')
        bing_endpoint = config.get('bing_endpoint', '')
        
        if not bing_api_key or not bing_endpoint:
            print("❌ Bing API key or endpoint not configured")
            return False
        
        print(f"🔑 Testing Bing API key: {bing_api_key[:8]}...")
        print(f"🌐 Endpoint: {bing_endpoint}")
        
        # Test search endpoint
        search_url = f"{bing_endpoint}/v7.0/search"
        headers = {'Ocp-Apim-Subscription-Key': bing_api_key}
        params = {
            'q': 'test query',
            'mkt': 'en-US',
            'count': 1,
            'responseFilter': ['Webpages']
        }
        
        print("🔍 Testing search endpoint...")
        response = requests.get(search_url, headers=headers, params=params)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'webPages' in result:
                print("✅ Bing Search API is working correctly!")
                return True
            else:
                print(f"⚠️  Unexpected response structure: {list(result.keys())}")
                return False
        elif response.status_code == 401:
            print("❌ Authentication failed - API key is invalid or expired")
            print(f"Response: {response.text}")
            return False
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Bing API: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Bing API Configuration...")
    success = test_bing_api()
    sys.exit(0 if success else 1)
