#!/usr/bin/env python3
"""
Test script to verify the psycopg deprecation warning fix.
"""

import asyncio
import os
import sys
import warnings

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_agent.utils.postgres_checkpointer import modern_psycopg_pool, cleanup_all_pools

async def test_modern_pool():
    """Test the modern psycopg pool approach without deprecation warnings."""
    print("Testing modern psycopg pool approach...")
    
    # Set up environment variables (you'll need to set these)
    if not all([os.environ.get(var) for var in ['host', 'port', 'dbname', 'user', 'password']]):
        print("⚠ Please set PostgreSQL environment variables: host, port, dbname, user, password")
        return False
    
    try:
        # Test the modern context manager approach
        async with modern_psycopg_pool() as pool:
            print("✓ Pool created successfully using modern approach")
            
            # Test getting a connection
            async with pool.connection() as conn:
                print("✓ Connection acquired from pool")
                
                # Test a simple query
                result = await conn.execute("SELECT 1 as test_value")
                row = await result.fetchone()
                print(f"✓ Query executed successfully: {row}")
        
        print("✓ Pool closed automatically via context manager")
        return True
        
    except Exception as e:
        print(f"✗ Error testing modern pool: {e}")
        return False

async def main():
    """Main test function."""
    print("Testing psycopg deprecation warning fix...")
    
    # Capture warnings to see if our fix worked
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        success = await test_modern_pool()
        
        # Check for deprecation warnings
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
        
        print(f"\nWarning Summary:")
        print(f"Deprecation warnings: {len(deprecation_warnings)}")
        print(f"Runtime warnings: {len(runtime_warnings)}")
        
        for warning in runtime_warnings:
            if "opening the async pool AsyncConnectionPool in the constructor is deprecated" in str(warning.message):
                print(f"⚠ Still getting deprecation warning: {warning.message}")
        
        if success and len(runtime_warnings) == 0:
            print("✅ SUCCESS: Modern approach works without deprecation warnings!")
        elif success:
            print("⚠ Pool works but still has warnings")
        else:
            print("❌ FAILED: Pool test failed")
    
    # Cleanup
    await cleanup_all_pools()
    print("🧹 Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())
