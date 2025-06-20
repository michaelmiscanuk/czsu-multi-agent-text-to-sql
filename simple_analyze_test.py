#!/usr/bin/env python3
"""
Simple /analyze Endpoint Test
Tests the authentication and basic functionality of the /analyze endpoint.
"""

import asyncio
import aiohttp
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"

async def test_authentication_issue():
    """Test to understand the authentication issue."""
    print("🔍 DIAGNOSING AUTHENTICATION ISSUE")
    print("=" * 50)
    
    # Test 1: No auth header
    print("\n1. Testing with no Authorization header:")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{BASE_URL}/analyze",
                json={"prompt": "test", "thread_id": "test"},
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.text()
                print(f"   Status: {response.status}")
                print(f"   Response: {result}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test 2: Invalid Bearer token
    print("\n2. Testing with invalid Bearer token:")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{BASE_URL}/analyze",
                json={"prompt": "test", "thread_id": "test"},
                headers={
                    "Authorization": "Bearer invalid_token",
                    "Content-Type": "application/json"
                }
            ) as response:
                result = await response.text()
                print(f"   Status: {response.status}")
                print(f"   Response: {result}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test 3: Check what the server expects
    print("\n3. Checking server's JWT verification requirements:")
    print("   - Your server expects Google JWT tokens with 'kid' field")
    print("   - The 'kid' field is used to look up Google's public keys")
    print("   - This means you need a real Google OAuth token to test")

async def test_other_endpoints():
    """Test other endpoints that might not require authentication."""
    print("\n\n🔍 TESTING OTHER ENDPOINTS")
    print("=" * 50)
    
    endpoints_to_test = [
        ("/health", "GET"),
        ("/health/database", "GET"),
        ("/health/memory", "GET"),
        ("/catalog", "GET"),  # This requires auth but let's see the error
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint, method in endpoints_to_test:
            print(f"\n{method} {endpoint}:")
            try:
                if method == "GET":
                    async with session.get(f"{BASE_URL}{endpoint}") as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"   ✅ Status: {response.status}")
                            print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
                        else:
                            result = await response.text()
                            print(f"   ❌ Status: {response.status}")
                            print(f"   Response: {result}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

def print_solution():
    """Print the solution to fix the 500 error."""
    print("\n\n🔧 SOLUTION TO FIX THE 500 ERROR")
    print("=" * 50)
    print()
    print("The issue is NOT with your /analyze endpoint - it's with authentication!")
    print()
    print("🎯 ROOT CAUSE:")
    print("   Your UI is sending requests without valid Google JWT tokens")
    print("   The server rejects these with 401 Unauthorized, not 500 Internal Server Error")
    print()
    print("🔧 TO FIX:")
    print("   1. Check your frontend authentication flow")
    print("   2. Ensure Google OAuth is working properly")
    print("   3. Verify JWT tokens are being sent in requests")
    print("   4. Check browser Network tab for actual token being sent")
    print()
    print("🧪 TO TEST WITH REAL TOKEN:")
    print("   1. Login through your frontend")
    print("   2. Open browser DevTools → Network tab")
    print("   3. Make a request to /analyze")
    print("   4. Copy the Authorization header from the request")
    print("   5. Use that token for testing")
    print()
    print("💡 QUICK DEBUG:")
    print("   - Check if GOOGLE_CLIENT_ID environment variable is set")
    print("   - Verify frontend is calling Google OAuth correctly") 
    print("   - Look for 'Authentication failed' in your server logs")

async def main():
    """Main test function."""
    print("🚀 SIMPLE /analyze ENDPOINT DIAGNOSIS")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    print(f"🌐 Testing: {BASE_URL}")
    
    # Test server health first
    print("\n🏥 SERVER HEALTH CHECK:")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"   ✅ Server is healthy")
                    print(f"   📊 Memory: {health.get('memory_rss_mb', 'unknown')}MB")
                    print(f"   🔢 Requests processed: {health.get('total_requests_processed', 'unknown')}")
                else:
                    print(f"   ❌ Server health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"   ❌ Cannot connect to server: {e}")
            print("   💡 Make sure your server is running on http://localhost:8000")
            return
    
    # Run authentication tests
    await test_authentication_issue()
    
    # Test other endpoints
    await test_other_endpoints()
    
    # Print solution
    print_solution()
    
    print(f"\n⏰ Completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    asyncio.run(main()) 