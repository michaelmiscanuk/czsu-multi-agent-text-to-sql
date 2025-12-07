"""Quick test script for Gemini integration"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is loaded
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {api_key[:20]}..." if api_key else "API Key NOT FOUND")

# Test Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.0,
        google_api_key=api_key,
    )
    print("✓ Gemini LLM instance created successfully!")

    # Test invocation
    print("\nSending test message...")
    response = llm.invoke("Say hello in one sentence")
    print(f"Response: {response.content}")
    print("\n✓ Test completed successfully!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
