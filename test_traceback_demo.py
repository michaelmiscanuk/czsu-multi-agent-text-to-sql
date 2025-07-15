#!/usr/bin/env python3
"""
Demo script to show the new server traceback capture functionality.
This script demonstrates how detailed server-side tracebacks are captured
and saved to files when testing API endpoints.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Set Windows event loop policy if needed
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import httpx
from tests.helpers import (
    make_request_with_traceback_capture,
    extract_detailed_error_info,
    save_server_traceback_report
)

async def demo_traceback_capture():
    """Demonstrate server traceback capture functionality."""
    print("🚀 Server Traceback Capture Demo")
    print("=" * 50)
    
    # This should trigger the NameError: db_pFDGDFGFDGath is not defined
    test_url = "http://localhost:8000/data-table?table=test_table"
    
    print(f"📡 Making request to: {test_url}")
    print("   This should trigger a server-side NameError due to the typo in catalog.py")
    print("   (db_pFDGDFGFDGath instead of db_path)")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create a test JWT token
        token = "test_token_placeholder"  # Simple token for demo
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            # Make request with traceback capture
            result = await make_request_with_traceback_capture(
                client,
                "GET",
                test_url,
                headers=headers
            )
            
            print(f"\n📊 Request completed:")
            print(f"   Success: {result['success']}")
            print(f"   Response: {result['response'].status_code if result['response'] else 'None'}")
            print(f"   Server logs captured: {len(result['server_logs'])}")
            print(f"   Server tracebacks captured: {len(result['server_tracebacks'])}")
            
            # Extract detailed error info
            error_info = extract_detailed_error_info(result)
            
            print(f"\n🔍 Error Analysis:")
            print(f"   Has server errors: {error_info['has_server_errors']}")
            print(f"   Has client errors: {error_info['has_client_errors']}")
            print(f"   HTTP status: {error_info['http_status']}")
            
            # Show server tracebacks
            if error_info['server_tracebacks']:
                print(f"\n📋 Server Tracebacks ({len(error_info['server_tracebacks'])}):")
                for i, tb in enumerate(error_info['server_tracebacks'], 1):
                    print(f"   {i}. {tb['exception_type']}: {tb['exception_message']}")
                    print(f"      Timestamp: {tb['timestamp']}")
                    print(f"      Level: {tb['level']}")
                    
                    # Show first few lines of traceback
                    tb_lines = tb['traceback'].split('\n')
                    print(f"      Traceback preview:")
                    for line in tb_lines[:5]:
                        if line.strip():
                            print(f"        {line}")
                    if len(tb_lines) > 5:
                        print(f"        ... ({len(tb_lines) - 5} more lines)")
                    print()
                
                # Save the traceback report
                print("💾 Saving traceback report...")
                save_server_traceback_report(
                    test_file_name="demo_traceback_test.py",
                    test_results=None,  # We don't have a full test results object
                    server_tracebacks=error_info['server_tracebacks'],
                    additional_info={
                        "Demo Type": "Server Traceback Capture",
                        "Test URL": test_url,
                        "Expected Error": "NameError: db_pFDGDFGFDGath is not defined"
                    }
                )
            else:
                print("\n⚠️  No server tracebacks captured. This could mean:")
                print("   - The server is not running")
                print("   - The error didn't occur as expected")
                print("   - The logging capture didn't work properly")
                
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Make sure your FastAPI server is running on http://localhost:8000")
    print("   The server should have the intentional typo in catalog.py")
    print("   (db_pFDGDFGFDGath instead of db_path)")
    print()
    
    try:
        asyncio.run(demo_traceback_capture())
    except KeyboardInterrupt:
        print("\n⛔ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc() 