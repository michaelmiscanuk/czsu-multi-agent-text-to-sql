#!/usr/bin/env python3
"""
Simple health monitoring script for the FastAPI application.
Can be run periodically to check app health and detect issues.
"""

import requests
import json
import time
from datetime import datetime

def check_health(base_url="http://localhost:10000"):
    """Check the health of the FastAPI application."""
    try:
        print(f"🔍 Checking health at {datetime.now().isoformat()}")
        
        # Check health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            status = health_data.get("status", "unknown")
            
            print(f"✅ Health check passed - Status: {status}")
            
            # Log memory usage if available
            if "memory_usage_mb" in health_data:
                memory_mb = health_data["memory_usage_mb"]
                memory_percent = health_data.get("memory_usage_percent")
                if memory_percent:
                    print(f"🧠 Memory: {memory_mb}MB ({memory_percent}%)")
                else:
                    print(f"🧠 Memory: {memory_mb}MB")
                
                # Warning for high memory usage
                if memory_percent and memory_percent > 80:
                    print(f"⚠️ HIGH MEMORY WARNING: {memory_percent}%")
            
            # Log database status
            if "database" in health_data:
                db_status = health_data["database"]
                print(f"💾 Database: {db_status}")
                
                if db_status != "connected":
                    print(f"⚠️ DATABASE WARNING: {db_status}")
            
            return True, health_data
            
        else:
            print(f"❌ Health check failed - Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print(f"⏰ Health check timed out")
        return False, None
    except requests.exceptions.ConnectionError:
        print(f"🔌 Connection error - app may be down")
        return False, None
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False, None

def monitor_continuously(interval_seconds=60, base_url="http://localhost:10000"):
    """Continuously monitor the application health."""
    print(f"🚀 Starting continuous health monitoring (interval: {interval_seconds}s)")
    print(f"📍 Monitoring URL: {base_url}")
    
    consecutive_failures = 0
    
    while True:
        try:
            success, health_data = check_health(base_url)
            
            if success:
                consecutive_failures = 0
                
                # Additional warnings based on health data
                if health_data:
                    status = health_data.get("status", "unknown")
                    if status in ["high_memory", "critical_memory", "degraded"]:
                        print(f"⚠️ APPLICATION WARNING: Status is '{status}'")
                        
            else:
                consecutive_failures += 1
                print(f"💥 Consecutive failures: {consecutive_failures}")
                
                if consecutive_failures >= 3:
                    print(f"🚨 CRITICAL: {consecutive_failures} consecutive health check failures!")
                    print(f"🚨 Application may have crashed or be unresponsive")
            
            print(f"---")
            time.sleep(interval_seconds)
            
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(interval_seconds)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "continuous":
            base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:10000"
            interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            monitor_continuously(interval, base_url)
        else:
            base_url = sys.argv[1]
            check_health(base_url)
    else:
        # Single health check
        check_health() 