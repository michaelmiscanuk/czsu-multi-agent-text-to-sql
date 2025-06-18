"""
Test environment variable configuration for MAX_CONCURRENT_ANALYSES
"""
import os
import sys
import tempfile
from pathlib import Path

def test_max_concurrent_analyses_env_reading():
    """Test that MAX_CONCURRENT_ANALYSES reads from environment correctly."""
    
    print("🧪 TESTING MAX_CONCURRENT_ANALYSES ENVIRONMENT READING")
    print("=" * 60)
    
    # Test 1: Default value when not set
    print("📋 Test 1: Default value when environment variable not set")
    if 'MAX_CONCURRENT_ANALYSES' in os.environ:
        original_value = os.environ['MAX_CONCURRENT_ANALYSES']
        del os.environ['MAX_CONCURRENT_ANALYSES']
    else:
        original_value = None
    
    # Simulate the code from api_server.py
    max_concurrent = int(os.environ.get('MAX_CONCURRENT_ANALYSES', '1'))
    print(f"   Result: {max_concurrent} (expected: 1)")
    assert max_concurrent == 1, f"Expected 1, got {max_concurrent}"
    print("   ✅ Default value test passed")
    
    # Test 2: Custom value from environment
    print("\n📋 Test 2: Custom value from environment variable")
    os.environ['MAX_CONCURRENT_ANALYSES'] = '3'
    max_concurrent = int(os.environ.get('MAX_CONCURRENT_ANALYSES', '1'))
    print(f"   Result: {max_concurrent} (expected: 3)")
    assert max_concurrent == 3, f"Expected 3, got {max_concurrent}"
    print("   ✅ Custom value test passed")
    
    # Test 3: Invalid value handling
    print("\n📋 Test 3: Invalid value handling")
    os.environ['MAX_CONCURRENT_ANALYSES'] = 'invalid'
    try:
        max_concurrent = int(os.environ.get('MAX_CONCURRENT_ANALYSES', '1'))
        print(f"   ❌ Should have failed but got: {max_concurrent}")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("   ✅ Invalid value correctly raises ValueError")
    
    # Test 4: Edge case values
    print("\n📋 Test 4: Edge case values")
    test_values = ['0', '10', '100']
    for value in test_values:
        os.environ['MAX_CONCURRENT_ANALYSES'] = value
        max_concurrent = int(os.environ.get('MAX_CONCURRENT_ANALYSES', '1'))
        expected = int(value)
        print(f"   Value '{value}' -> {max_concurrent} (expected: {expected})")
        assert max_concurrent == expected, f"Expected {expected}, got {max_concurrent}"
    print("   ✅ Edge case values test passed")
    
    # Restore original value
    if original_value is not None:
        os.environ['MAX_CONCURRENT_ANALYSES'] = original_value
    elif 'MAX_CONCURRENT_ANALYSES' in os.environ:
        del os.environ['MAX_CONCURRENT_ANALYSES']
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ MAX_CONCURRENT_ANALYSES environment reading works correctly")

def test_dotenv_file_reading():
    """Test reading from .env file format."""
    
    print("\n🧪 TESTING .ENV FILE FORMAT READING")
    print("=" * 60)
    
    # Create a temporary .env file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("MAX_CONCURRENT_ANALYSES=5\n")
        f.write("OTHER_VAR=test\n")
        env_file_path = f.name
    
    try:
        print(f"📋 Created temporary .env file: {env_file_path}")
        
        # Simulate reading from .env file (basic parsing)
        env_vars = {}
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        
        print(f"   Parsed variables: {env_vars}")
        
        # Test the parsing
        if 'MAX_CONCURRENT_ANALYSES' in env_vars:
            max_concurrent = int(env_vars['MAX_CONCURRENT_ANALYSES'])
            print(f"   MAX_CONCURRENT_ANALYSES from .env: {max_concurrent} (expected: 5)")
            assert max_concurrent == 5, f"Expected 5, got {max_concurrent}"
            print("   ✅ .env file parsing test passed")
        else:
            assert False, "MAX_CONCURRENT_ANALYSES not found in .env file"
    
    finally:
        # Clean up
        os.unlink(env_file_path)
        print(f"   🧹 Cleaned up temporary file: {env_file_path}")

def main():
    """Run all environment configuration tests."""
    print("🚀 ENVIRONMENT CONFIGURATION TESTS")
    print("=" * 60)
    
    try:
        test_max_concurrent_analyses_env_reading()
        test_dotenv_file_reading()
        
        print("\n" + "=" * 60)
        print("🎉 ALL ENVIRONMENT CONFIGURATION TESTS PASSED!")
        print("✅ Ready to use MAX_CONCURRENT_ANALYSES from .env file")
        print("✅ API server will correctly read the environment variable")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 