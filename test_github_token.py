"""Test script to verify GitHub token permissions and model limits."""

import os
from dotenv import load_dotenv
import requests

load_dotenv()

github_token = os.getenv("GITHUB_TOKEN")

if not github_token:
    print("❌ GITHUB_TOKEN not found in environment variables")
    exit(1)

print("🔍 Checking GitHub token authentication...\n")

# Check token authentication and user info
headers = {
    "Authorization": f"Bearer {github_token}",
    "Accept": "application/vnd.github+json",
}

# Get user info
response = requests.get("https://api.github.com/user", headers=headers)
if response.status_code == 200:
    user_data = response.json()
    print(f"✅ Authenticated as: {user_data.get('login')}")
    print(f"   User type: {user_data.get('type')}")
    print(f"   Plan: {user_data.get('plan', {}).get('name', 'N/A')}")
else:
    print(f"❌ Authentication failed: {response.status_code}")
    print(f"   Response: {response.text}")
    exit(1)

# Check organizations
print("\n🏢 Checking organizations...")
response = requests.get("https://api.github.com/user/orgs", headers=headers)
if response.status_code == 200:
    orgs = response.json()
    if orgs:
        print(f"   Member of {len(orgs)} organization(s):")
        for org in orgs:
            print(f"   - {org.get('login')}")
    else:
        print("   ⚠️  Not a member of any organizations")
        print("   ⚠️  You may not have Enterprise tier access")
else:
    print(f"   ⚠️  Could not fetch organizations: {response.status_code}")

# Check token scopes
print("\n🔐 Token scopes:")
scopes_header = response.headers.get("X-OAuth-Scopes", "")
if scopes_header:
    scopes = [s.strip() for s in scopes_header.split(",")]
    for scope in scopes:
        print(f"   - {scope}")
    if "models:read" in scopes or "read:org" in scopes:
        print("   ✅ Has models access")
    else:
        print("   ⚠️  May not have 'models:read' scope")
else:
    print("   ⚠️  Could not determine scopes (might be a fine-grained PAT)")

# Test a simple API call to GitHub Models
print("\n🧪 Testing GitHub Models API with DeepSeek-V3-0324...")
from langchain_openai import ChatOpenAI

try:
    llm = ChatOpenAI(
        model="deepseek/DeepSeek-V3-0324",
        api_key=github_token,
        base_url="https://models.github.ai/inference",
    )
    
    # Send a very short message
    response = llm.invoke("Hi")
    print(f"✅ API call successful!")
    print(f"   Response: {response.content[:100]}")
    
except Exception as e:
    print(f"❌ API call failed: {str(e)}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nIf you see '⚠️ Not a member of any organizations' above,")
print("you're likely using a personal account without Enterprise access.")
print("\nTo get Enterprise tier limits (16000 in / 8000 out), you need:")
print("1. GitHub Enterprise organization membership")
print("2. Organization must have GitHub Models Enterprise tier")
print("3. Token created through that organization")
print("\nAlternatively, consider:")
print("- Use a different model with higher free tier limits (e.g., GPT-4o)")
print("- Opt in to paid GitHub Models")
print("- Use DeepSeek through their direct API or OpenRouter")
