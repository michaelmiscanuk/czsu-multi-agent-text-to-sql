# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
import os
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Constants
try:
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import traceback
from fastapi import Header, HTTPException

# Import JWT verification function
from api.auth.jwt_auth import verify_google_jwt

# Import debug utilities
from api.utils.debug import print__token_debug
from api.utils.memory import log_comprehensive_error

#============================================================
# AUTHENTICATION DEPENDENCIES
#============================================================
# Enhanced dependency for JWT authentication with better error handling
def get_current_user(authorization: str = Header(None)):
    try:
        # Add debug prints using the user's enabled environment variables
        print__token_debug("🔑 AUTHENTICATION START: Beginning user authentication process")
        print__token_debug("🔑 AUTH TRACE: get_current_user called with authorization header")
        
        if not authorization:
            print__token_debug("❌ AUTH ERROR: No authorization header provided")
            print__token_debug("❌ AUTH TRACE: Missing Authorization header - raising 401")
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        
        print__token_debug(f"🔍 AUTH CHECK: Authorization header present (length: {len(authorization)})")
        print__token_debug(f"🔍 AUTH TRACE: Authorization header format check - starts with 'Bearer ': {authorization.startswith('Bearer ')}")
        
        if not authorization.startswith("Bearer "):
            print__token_debug("❌ AUTH ERROR: Invalid authorization header format")
            print__token_debug("❌ AUTH TRACE: Invalid Authorization header format - raising 401")
            raise HTTPException(status_code=401, detail="Invalid Authorization header format. Expected 'Bearer <token>'")
        
        # Split and validate token extraction
        auth_parts = authorization.split(" ", 1)
        if len(auth_parts) != 2 or not auth_parts[1].strip():
            print__token_debug("❌ AUTH ERROR: Malformed authorization header")
            print__token_debug(f"❌ AUTH TRACE: Authorization header split failed - parts: {len(auth_parts)}")
            raise HTTPException(status_code=401, detail="Invalid Authorization header format")
        
        token = auth_parts[1].strip()
        print__token_debug(f"🔍 AUTH TOKEN: Token extracted successfully (length: {len(token)})")
        print__token_debug(f"🔍 AUTH TRACE: Token validation starting with verify_google_jwt")
        
        # Call JWT verification with debug tracing
        user_info = verify_google_jwt(token)
        print__token_debug(f"✅ AUTH SUCCESS: User authenticated successfully - {user_info.get('email', 'Unknown')}")
        print__token_debug(f"✅ AUTH TRACE: verify_google_jwt returned user info: {user_info}")
        
        return user_info
        
    except HTTPException as he:
        # Re-raise HTTPException with enhanced debugging
        print__token_debug(f"❌ AUTH HTTP EXCEPTION: {he.status_code} - {he.detail}")
        print__token_debug(f"❌ AUTH TRACE: HTTPException caught - status: {he.status_code}, detail: {he.detail}")
        print__token_debug(f"❌ AUTH TRACE: Full HTTPException traceback:\n{traceback.format_exc()}")
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        # Enhanced error handling with full traceback
        print__token_debug(f"❌ AUTH EXCEPTION: Unexpected authentication error - {type(e).__name__}: {str(e)}")
        print__token_debug(f"❌ AUTH TRACE: Unexpected exception in authentication")
        print__token_debug(f"❌ AUTH TRACE: Full traceback:\n{traceback.format_exc()}")
        
        print__token_debug(f"Authentication error: {e}")
        log_comprehensive_error("authentication", e)
        raise HTTPException(status_code=401, detail="Authentication failed") 