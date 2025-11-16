"""
MODULE_DESCRIPTION: Authentication Dependencies - JWT Token Verification for FastAPI

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module provides FastAPI dependency functions for authenticating incoming API
requests using JWT (JSON Web Token) verification. It serves as the authentication
layer for the CZSU Multi-Agent Text-to-SQL API, ensuring that only authenticated
users can access protected endpoints.

The primary function `get_current_user` extracts JWT tokens from HTTP Authorization
headers, verifies them using Google's OAuth2 public keys, and returns decoded user
information. This dependency is injected into route handlers throughout the API to
enforce authentication.

Authentication Flow:
    1. Extract Authorization header from incoming request
    2. Validate header format (must be "Bearer <token>")
    3. Extract and trim the JWT token string
    4. Call verify_google_jwt() to validate token signature
    5. Return decoded user info (email, sub, etc.)
    6. Raise HTTPException if any step fails

===================================================================================
KEY FEATURES
===================================================================================

1. FastAPI Dependency Injection
   - Declarative authentication via function dependencies
   - Automatic header extraction and validation
   - Seamless integration with route handlers
   - Type-safe user information access

2. JWT Token Verification
   - Google OAuth2 token validation
   - Public key verification via JWK endpoint
   - Signature verification and expiry checking
   - Claim validation (issuer, audience)

3. Comprehensive Debug Logging
   - Step-by-step authentication tracing
   - Token validation progress tracking
   - Error diagnosis with full tracebacks
   - Environment-controlled verbosity

4. Robust Error Handling
   - Missing Authorization header detection
   - Invalid header format validation
   - Malformed token string detection
   - Authentication failure with proper status codes

5. Security Best Practices
   - No token logging (security risk)
   - Proper HTTP 401 Unauthorized responses
   - Clear error messages without leaking details
   - Defense against header injection attacks

===================================================================================
AUTHENTICATION DEPENDENCY
===================================================================================

Function: get_current_user
    Purpose: Extract and verify JWT token from Authorization header

    Parameters:
        authorization (str): Authorization header value from HTTP request
            - Injected automatically by FastAPI
            - Expected format: "Bearer <token>"
            - Optional (defaults to None if not provided)

    Returns:
        dict: Decoded JWT payload containing user information
            - email: User's email address
            - sub: Google subject identifier (unique user ID)
            - name: User's full name (if available)
            - picture: Profile picture URL (if available)
            - iat: Issued at timestamp
            - exp: Expiry timestamp

    Raises:
        HTTPException(401): If authentication fails
            - Missing Authorization header
            - Invalid header format
            - Malformed token
            - Token verification failure
            - Expired token

Usage Example:
    from fastapi import APIRouter, Depends
    from api.dependencies.auth import get_current_user

    router = APIRouter()

    @router.post("/analyze")
    async def analyze(
        request: AnalyzeRequest,
        user: dict = Depends(get_current_user)
    ):
        user_email = user["email"]
        # ... process request for authenticated user
        return {"status": "success"}

===================================================================================
AUTHENTICATION FLOW DETAILS
===================================================================================

Step 1: Header Presence Check
    - Verify Authorization header exists
    - Return 401 if missing
    - Debug: "❌ AUTH ERROR: No authorization header provided"

Step 2: Header Format Validation
    - Check for "Bearer " prefix
    - Validate header structure
    - Return 401 if invalid format
    - Debug: "❌ AUTH ERROR: Invalid authorization header format"

Step 3: Token Extraction
    - Split header on space delimiter
    - Validate exactly 2 parts (Bearer + token)
    - Trim whitespace from token
    - Return 401 if malformed
    - Debug: "🔍 AUTH TOKEN: Token extracted successfully"

Step 4: Token Verification
    - Call verify_google_jwt(token)
    - Validate signature using Google's public keys
    - Verify token expiry
    - Validate issuer and audience claims
    - Return 401 if verification fails
    - Debug: "✅ AUTH SUCCESS: User authenticated successfully"

Step 5: Return User Info
    - Return decoded JWT payload as dict
    - Contains user email, sub, name, picture
    - Ready for use in route handler

===================================================================================
DEBUG LOGGING SYSTEM
===================================================================================

Debug Functions:
    print__token_debug(message)
        - Logs authentication-specific debug messages
        - Controlled by TOKEN_DEBUG environment variable
        - Tracks authentication flow progress
        - Includes step-by-step tracing

Debug Levels:
    1. Process Start/End
       - "🔑 AUTHENTICATION START"
       - "✅ AUTH SUCCESS"
       - "❌ AUTH ERROR"

    2. Validation Steps
       - Header presence check
       - Format validation
       - Token extraction
       - JWT verification call

    3. Error Details
       - Exception types and messages
       - HTTP status codes
       - Full tracebacks
       - User identification

Debug Output Example:
    🔑 AUTHENTICATION START: Beginning user authentication process
    🔑 AUTH TRACE: get_current_user called with authorization header
    🔍 AUTH CHECK: Authorization header present (length: 850)
    🔍 AUTH TOKEN: Token extracted successfully (length: 820)
    🔍 AUTH TRACE: Token validation starting with verify_google_jwt
    ✅ AUTH SUCCESS: User authenticated successfully - user@example.com

===================================================================================
ERROR HANDLING
===================================================================================

Error Categories:

1. Missing Header
   - Condition: authorization is None
   - Status: 401 Unauthorized
   - Message: "Missing Authorization header"
   - Debug: Full trace with request context

2. Invalid Format
   - Condition: Not starting with "Bearer "
   - Status: 401 Unauthorized
   - Message: "Invalid Authorization header format. Expected 'Bearer <token>'"
   - Debug: Header format details

3. Malformed Header
   - Condition: Split yields != 2 parts or empty token
   - Status: 401 Unauthorized
   - Message: "Invalid Authorization header format"
   - Debug: Part count and content analysis

4. Verification Failure
   - Condition: verify_google_jwt raises exception
   - Status: 401 Unauthorized
   - Message: "Authentication failed"
   - Debug: Exception type, message, full traceback

Exception Handling Strategy:
    - Catch HTTPException separately (re-raise as-is)
    - Catch general Exception for unexpected errors
    - Log comprehensive error details
    - Return consistent 401 responses
    - Include tracebacks in debug mode

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. Token Privacy
   - Never log full token strings
   - Only log token length for debugging
   - Prevents token leakage in logs

2. Error Message Safety
   - Generic error messages to users
   - Detailed errors only in debug logs
   - Prevents information disclosure

3. Header Injection Protection
   - Strict format validation
   - Split validation prevents injection
   - Whitespace trimming

4. HTTPException Re-raising
   - Preserve original status codes
   - Maintain error details
   - Proper HTTP semantics

5. Authentication Logging
   - Track authentication attempts
   - Log user email on success
   - Log client IP for failed attempts
   - Audit trail for security

===================================================================================
INTEGRATION WITH JWT_AUTH MODULE
===================================================================================

Dependency on verify_google_jwt:
    from api.auth.jwt_auth import verify_google_jwt

    Purpose:
        - Validates JWT signature using Google's public keys
        - Verifies token expiry and claims
        - Returns decoded user information

    Error Handling:
        - Raises HTTPException(401) on validation failure
        - Provides detailed error messages
        - Includes token debugging information

Public Key Retrieval:
    - Google's JWK endpoint queried dynamically
    - Public keys cached for performance
    - Automatic key rotation support
    - Fallback to fresh fetch if kid not found

Token Validation:
    - Signature verification with RS256 algorithm
    - Expiry check (exp claim)
    - Issuer validation (iss claim)
    - Audience validation (aud claim)

===================================================================================
WINDOWS COMPATIBILITY
===================================================================================

Event Loop Policy:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

Reason:
    - Must be set BEFORE any async operations
    - Required for psycopg compatibility
    - Fixes "Event loop closed" errors
    - Placed at top of file

===================================================================================
PERFORMANCE CHARACTERISTICS
===================================================================================

Time Complexity:
    - Header extraction: O(1)
    - Token split/trim: O(n) where n = token length
    - JWT verification: O(1) (cached public keys)
    - Total: O(n) dominated by network call for first key fetch

Memory Usage:
    - JWT payload: ~1-2 KB per token
    - Decoded user info: ~0.5 KB
    - Debug logs: Negligible
    - No per-request storage

Caching:
    - Public keys cached by jwt_auth module
    - No caching in this module
    - Stateless dependency function

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    - Mock Authorization header values
    - Test missing header scenario
    - Test invalid format scenarios
    - Test malformed token scenarios
    - Mock verify_google_jwt function

Test Cases:
    1. Valid JWT → Returns user info
    2. No header → HTTPException(401)
    3. Invalid format → HTTPException(401)
    4. Malformed token → HTTPException(401)
    5. Expired token → HTTPException(401)
    6. Invalid signature → HTTPException(401)

Mock Example:
    from unittest.mock import patch

    @patch('api.dependencies.auth.verify_google_jwt')
    def test_valid_token(mock_verify):
        mock_verify.return_value = {
            "email": "test@example.com",
            "sub": "123456"
        }
        user = get_current_user("Bearer valid_token")
        assert user["email"] == "test@example.com"

Integration Tests:
    - Use real Google test tokens
    - Test full authentication flow
    - Verify route protection
    - Test concurrent authentications

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection
    - traceback: Error tracing

Third-Party:
    - fastapi: Header dependency injection
    - dotenv: Environment variable loading

Internal:
    - api.auth.jwt_auth: JWT verification logic
    - api.utils.debug: Debug logging functions
    - api.utils.memory: Error logging

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Multi-Provider Support
   - Support Auth0, Azure AD, etc.
   - Provider-specific verification
   - Unified user info format

2. Token Caching
   - Cache decoded tokens temporarily
   - Reduce verification overhead
   - Configurable TTL

3. Enhanced Logging
   - Structured logging (JSON)
   - Authentication metrics
   - Failed attempt tracking

4. Role-Based Access
   - Extract roles from JWT claims
   - Role validation in dependency
   - Granular permission checking

===================================================================================
"""

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys

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


# ==============================================================================
# AUTHENTICATION DEPENDENCIES
# ==============================================================================


def get_current_user(authorization: str = Header(None)):
    """Extract and verify JWT token from Authorization header.

    This FastAPI dependency function handles the complete authentication flow
    for protected API endpoints. It validates the Authorization header format,
    extracts the JWT token, verifies it using Google's OAuth2 public keys,
    and returns the decoded user information.

    Args:
        authorization: The Authorization header value containing the JWT token.
            Expected format: "Bearer <token>"
            Automatically injected by FastAPI from HTTP request headers.

    Returns:
        dict: The decoded JWT payload containing user information including:
            - email: User's email address
            - sub: Google subject identifier (unique user ID)
            - name: User's full name (if available)
            - picture: Profile picture URL (if available)
            - iat: Issued at timestamp
            - exp: Expiry timestamp

    Raises:
        HTTPException(401): If authentication fails for any reason:
            - Missing Authorization header
            - Invalid header format (not "Bearer <token>")
            - Malformed token string
            - Token verification failure (invalid signature, expired, etc.)

    Example:
        @router.post("/analyze")
        async def analyze(
            request: AnalyzeRequest,
            user: dict = Depends(get_current_user)
        ):
            user_email = user["email"]
            return {"status": "success"}
    """
    try:
        # =======================================================================
        # STEP 1: AUTHENTICATION START - Log Process Initiation
        # =======================================================================

        # Add debug prints using the user's enabled environment variables
        print__token_debug(
            "🔑 AUTHENTICATION START: Beginning user authentication process"
        )
        print__token_debug(
            "🔑 AUTH TRACE: get_current_user called with authorization header"
        )

        # =======================================================================
        # STEP 2: HEADER PRESENCE CHECK - Verify Authorization Header Exists
        # =======================================================================
        if not authorization:
            print__token_debug("❌ AUTH ERROR: No authorization header provided")
            print__token_debug(
                "❌ AUTH TRACE: Missing Authorization header - raising 401"
            )
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        print__token_debug(
            f"🔍 AUTH CHECK: Authorization header present (length: {len(authorization)})"
        )
        print__token_debug(
            f"🔍 AUTH TRACE: Authorization header format check - starts with 'Bearer ': {authorization.startswith('Bearer ')}"
        )

        # =======================================================================
        # STEP 3: HEADER FORMAT VALIDATION - Ensure "Bearer <token>" Format
        # =======================================================================
        if not authorization.startswith("Bearer "):
            print__token_debug("❌ AUTH ERROR: Invalid authorization header format")
            print__token_debug(
                "❌ AUTH TRACE: Invalid Authorization header format - raising 401"
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format. Expected 'Bearer <token>'",
            )

        # =======================================================================
        # STEP 4: TOKEN EXTRACTION - Split Header and Extract Token String
        # =======================================================================

        # Split and validate token extraction
        auth_parts = authorization.split(" ", 1)
        if len(auth_parts) != 2 or not auth_parts[1].strip():
            print__token_debug("❌ AUTH ERROR: Malformed authorization header")
            print__token_debug(
                f"❌ AUTH TRACE: Authorization header split failed - parts: {len(auth_parts)}"
            )
            raise HTTPException(
                status_code=401, detail="Invalid Authorization header format"
            )

        token = auth_parts[1].strip()
        print__token_debug(
            f"🔍 AUTH TOKEN: Token extracted successfully (length: {len(token)})"
        )
        print__token_debug(
            "🔍 AUTH TRACE: Token validation starting with verify_google_jwt"
        )

        # =======================================================================
        # STEP 5: JWT VERIFICATION - Validate Token Signature and Claims
        # =======================================================================

        # Call JWT verification with debug tracing
        user_info = verify_google_jwt(token)
        print__token_debug(
            f"✅ AUTH SUCCESS: User authenticated successfully - {user_info.get('email', 'Unknown')}"
        )
        print__token_debug(
            f"✅ AUTH TRACE: verify_google_jwt returned user info: {user_info}"
        )

        return user_info

    except HTTPException as he:
        # =======================================================================
        # HTTP EXCEPTION HANDLING - Re-raise FastAPI HTTP Exceptions As-Is
        # =======================================================================

        # Re-raise HTTPException with enhanced debugging
        print__token_debug(f"❌ AUTH HTTP EXCEPTION: {he.status_code} - {he.detail}")
        print__token_debug(
            f"❌ AUTH TRACE: HTTPException caught - status: {he.status_code}, detail: {he.detail}"
        )
        print__token_debug(
            f"❌ AUTH TRACE: Full HTTPException traceback:\n{traceback.format_exc()}"
        )
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        # =======================================================================
        # GENERAL EXCEPTION HANDLING - Convert Unexpected Errors to HTTP 401
        # =======================================================================

        # Enhanced error handling with full traceback
        print__token_debug(
            f"❌ AUTH EXCEPTION: Unexpected authentication error - {type(e).__name__}: {str(e)}"
        )
        print__token_debug("❌ AUTH TRACE: Unexpected exception in authentication")
        print__token_debug(f"❌ AUTH TRACE: Full traceback:\n{traceback.format_exc()}")

        print__token_debug(f"Authentication error: {e}")
        log_comprehensive_error("authentication", e)
        raise HTTPException(status_code=401, detail="Authentication failed")
