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

import time
import traceback

# Standard imports
import jwt
import requests
from fastapi import HTTPException
from jwt.algorithms import RSAAlgorithm
from jwt.exceptions import ImmatureSignatureError

# Import constants from api.config.settings
from api.config.settings import GOOGLE_JWK_URL, _jwt_kid_missing_count

# Import debug utilities
from api.utils.debug import print__token_debug


# ============================================================
# AUTHENTICATION - JWT VERIFICATION
# ============================================================
def verify_google_jwt(token: str):
    global _jwt_kid_missing_count

    try:
        # SYSTEM TIME CHECK: Warn if system time appears to be incorrect
        current_time = time.time()
        current_date = time.ctime(current_time)
        print__token_debug(
            f"🕐 SYSTEM TIME CHECK: Current system time is {current_date}"
        )

        # Check if we're in a reasonable time range (2024-2026)
        import datetime

        current_year = datetime.datetime.now().year
        if current_year < 2024 or current_year > 2026:
            print__token_debug(
                f"⚠️  WARNING: System year {current_year} seems incorrect for token validation"
            )
            print__token_debug(
                "This may cause JWT validation issues due to timestamp mismatches"
            )

        # EARLY VALIDATION: Check if token has basic JWT structure before processing
        # JWT tokens must have exactly 3 parts separated by dots (header.payload.signature)
        token_parts = token.split(".")
        if len(token_parts) != 3:
            # Don't log this as it's a common case for invalid tokens in tests
            raise HTTPException(status_code=401, detail="Invalid JWT token format")

        # Additional basic validation - each part should be non-empty and base64-like
        for i, part in enumerate(token_parts):
            if (
                not part or len(part) < 4
            ):  # Base64 encoded parts should be at least 4 chars
                raise HTTPException(status_code=401, detail="Invalid JWT token format")

        # Get unverified header and payload first - this should now work since we pre-validated the format
        try:
            unverified_header = jwt.get_unverified_header(token)
            # Add leeway for unverified payload decode as well to handle clock sync issues
            unverified_payload = jwt.decode(
                token, options={"verify_signature": False}, leeway=300
            )
        except jwt.DecodeError as e:
            # This should be rare now due to pre-validation, but keep for edge cases
            print__token_debug(f"JWT decode error after pre-validation: {e}")
            raise HTTPException(status_code=401, detail="Invalid JWT token format")
        except Exception as e:
            print__token_debug(f"JWT header decode error: {e}")
            raise HTTPException(status_code=401, detail="Invalid JWT token format")

        # Debug: print the audience in the token and the expected audience
        print__token_debug(f"Token aud: {unverified_payload.get('aud')}")
        print__token_debug(f"Backend GOOGLE_CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID')}")
        print__token_debug(f"Token iss: {unverified_payload.get('iss')}")

        # Debug: Print token timing information to help diagnose clock sync issues
        token_iat = unverified_payload.get("iat")
        token_exp = unverified_payload.get("exp")
        current_time = time.time()
        if token_iat:
            print__token_debug(
                f"Token iat (issued at): {token_iat} ({time.ctime(token_iat)})"
            )
        if token_exp:
            print__token_debug(
                f"Token exp (expires): {token_exp} ({time.ctime(token_exp)})"
            )
        print__token_debug(
            f"Current server time: {current_time} ({time.ctime(current_time)})"
        )
        if token_iat:
            time_diff = current_time - token_iat
            print__token_debug(
                f"Time difference (server - token_iat): {time_diff:.2f} seconds"
            )
            if time_diff < 0:
                print__token_debug(
                    f"⚠️  WARNING: Token issued {abs(time_diff):.2f} seconds in the future!"
                )
                print__token_debug(
                    "This indicates a clock synchronization issue - adding 5-minute leeway should resolve this"
                )
                # If time difference is more than 5 minutes, log it as severe
                if abs(time_diff) > 300:
                    print__token_debug(
                        f"🚨 SEVERE CLOCK SYNC ISSUE: Token is {abs(time_diff):.2f} seconds in the future (>{300}s leeway)"
                    )

        # TEST MODE: Handle test tokens with test issuer (for development/testing only)

        print__token_debug(
            f"🔍 ENV DEBUG: USE_TEST_TOKENS = '{os.getenv('USE_TEST_TOKENS', 'NOT_SET')}'"
        )
        # Only enabled when USE_TEST_TOKENS environment variable is set to "1"
        use_test_tokens = os.getenv("USE_TEST_TOKENS", "0") == "1"
        if use_test_tokens and unverified_payload.get("iss") == "test_issuer":
            print__token_debug(
                "🧪 TEST MODE: Detected test token with test issuer - skipping Google verification"
            )
            print__token_debug(
                f"🧪 TEST MODE: USE_TEST_TOKENS={os.getenv('USE_TEST_TOKENS', '0')} - test tokens enabled"
            )

            # Verify the audience still matches
            expected_aud = os.getenv("GOOGLE_CLIENT_ID")
            if unverified_payload.get("aud") != expected_aud:
                print__token_debug(
                    f"Test token audience mismatch. Expected: {expected_aud}, Got: {unverified_payload.get('aud')}"
                )
                raise HTTPException(
                    status_code=401, detail="Invalid test token audience"
                )

            # Check expiration
            if int(unverified_payload.get("exp", 0)) < time.time():
                print__token_debug("Test token has expired")
                raise HTTPException(status_code=401, detail="Test token has expired")

            # Return the test payload directly (no Google verification needed)
            print__token_debug("✅ TEST MODE: Test token validation successful")
            return unverified_payload
        elif unverified_payload.get("iss") == "test_issuer":
            # Test token detected but test mode is disabled
            print__token_debug(
                f"🚫 TEST MODE DISABLED: Test token detected but USE_TEST_TOKENS={os.getenv('USE_TEST_TOKENS', '0')} - rejecting token"
            )
            raise HTTPException(
                status_code=401,
                detail="Test tokens are not allowed in this environment",
            )

        # NEW: Check if this is a NextAuth.js id_token (missing 'kid' field)
        if "kid" not in unverified_header:
            # Reduce log noise - only log this every 10th occurrence
            _jwt_kid_missing_count += 1
            if _jwt_kid_missing_count % 10 == 1:  # Log 1st, 11th, 21st, etc.
                print__token_debug(
                    f"JWT token missing 'kid' field (#{_jwt_kid_missing_count}) - attempting NextAuth.js id_token verification"
                )

            # NEXTAUTH.JS SUPPORT: Verify id_token directly using Google's tokeninfo endpoint
            try:
                print__token_debug(
                    "Attempting NextAuth.js id_token verification via Google tokeninfo endpoint"
                )

                # Use Google's tokeninfo endpoint to verify the id_token
                tokeninfo_url = (
                    f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
                )
                response = requests.get(tokeninfo_url, timeout=10)

                if response.status_code == 200:
                    tokeninfo = response.json()
                    print__token_debug(f"Google tokeninfo response: {tokeninfo}")

                    # Verify the audience matches our client ID
                    expected_aud = os.getenv("GOOGLE_CLIENT_ID")
                    if tokeninfo.get("aud") != expected_aud:
                        print__token_debug(
                            f"Tokeninfo audience mismatch. Expected: {expected_aud}, Got: {tokeninfo.get('aud')}"
                        )
                        raise HTTPException(
                            status_code=401, detail="Invalid token audience"
                        )

                    # Verify the token is not expired
                    if int(tokeninfo.get("exp", 0)) < time.time():
                        print__token_debug("Tokeninfo shows token has expired")
                        raise HTTPException(status_code=401, detail="Token has expired")

                    # Return the tokeninfo as the payload (it contains email, name, etc.)
                    print__token_debug(
                        "NextAuth.js id_token verification successful via Google tokeninfo"
                    )
                    return tokeninfo

                else:
                    print__token_debug(
                        f"Google tokeninfo endpoint returned error: {response.status_code} - {response.text}"
                    )
                    raise HTTPException(
                        status_code=401, detail="Invalid NextAuth.js id_token"
                    )

            except requests.RequestException as e:
                print__token_debug(
                    f"Failed to verify NextAuth.js id_token via Google tokeninfo: {e}"
                )
                raise HTTPException(
                    status_code=401,
                    detail="Token verification failed - unable to validate NextAuth.js token",
                )
            except HTTPException:
                raise  # Re-raise HTTPException as-is
            except Exception as e:
                print__token_debug(f"NextAuth.js id_token verification failed: {e}")
                raise HTTPException(
                    status_code=401, detail="NextAuth.js token verification failed"
                )

        # ORIGINAL FLOW: Standard Google JWT token with 'kid' field (for direct Google API calls)
        try:
            # Get Google public keys for JWKS verification
            jwks = requests.get(GOOGLE_JWK_URL).json()
        except requests.RequestException as e:
            print__token_debug(f"Failed to fetch Google JWKS: {e}")
            raise HTTPException(
                status_code=401,
                detail="Token verification failed - unable to fetch Google keys",
            )

        # Find matching key
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                public_key = RSAAlgorithm.from_jwk(key)
                try:
                    # Add leeway to handle clock synchronization issues
                    # This allows tokens to be valid even if there's a small time difference
                    # between Google's servers and our local machine (up to 300 seconds / 5 minutes)
                    payload = jwt.decode(
                        token,
                        public_key,
                        algorithms=["RS256"],
                        audience=os.getenv("GOOGLE_CLIENT_ID"),
                        leeway=300,  # Allow 300 seconds (5 minutes) leeway for clock synchronization
                    )
                    print__token_debug(
                        "Standard Google JWT token verification successful"
                    )
                    return payload
                except jwt.ExpiredSignatureError:
                    print__token_debug("JWT token has expired")
                    raise HTTPException(status_code=401, detail="Token has expired")
                except jwt.ImmatureSignatureError as e:
                    print__token_debug(
                        f"JWT token not yet valid (clock sync issue): {e}"
                    )
                    print__token_debug(
                        "This usually indicates a clock synchronization problem between your system and Google's servers"
                    )
                    print__token_debug(
                        "Even with 5-minute leeway, token is still considered immature"
                    )
                    # Log more detailed timing information
                    if token_iat:
                        time_diff = current_time - token_iat
                        print__token_debug(
                            f"Detailed timing: Server time - Token iat = {time_diff:.2f} seconds"
                        )
                        if time_diff < -300:
                            print__token_debug(
                                f"Token is issued {abs(time_diff):.2f} seconds in the future, beyond 5-minute leeway"
                            )

                    # Try to continue with NextAuth.js flow as fallback
                    print__token_debug(
                        "Attempting fallback to NextAuth.js id_token verification"
                    )
                    try:
                        tokeninfo_url = (
                            f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
                        )
                        response = requests.get(tokeninfo_url, timeout=10)

                        if response.status_code == 200:
                            tokeninfo = response.json()
                            print__token_debug(
                                f"Google tokeninfo fallback successful: {tokeninfo.get('email', 'unknown')}"
                            )

                            # Verify the audience matches our client ID
                            expected_aud = os.getenv("GOOGLE_CLIENT_ID")
                            if tokeninfo.get("aud") != expected_aud:
                                print__token_debug(
                                    f"Tokeninfo audience mismatch. Expected: {expected_aud}, Got: {tokeninfo.get('aud')}"
                                )
                                raise HTTPException(
                                    status_code=401, detail="Invalid token audience"
                                )

                            # Return the tokeninfo as the payload
                            print__token_debug(
                                "Clock sync issue resolved via Google tokeninfo fallback"
                            )
                            return tokeninfo
                        else:
                            print__token_debug(
                                f"Google tokeninfo fallback failed: {response.status_code}"
                            )
                    except Exception as fallback_e:
                        print__token_debug(f"Tokeninfo fallback failed: {fallback_e}")

                    # If fallback fails, raise the original error
                    raise HTTPException(
                        status_code=401,
                        detail="Token not yet valid - severe clock synchronization issue detected",
                    )
                except jwt.InvalidAudienceError:
                    print__token_debug("JWT token has invalid audience")
                    raise HTTPException(
                        status_code=401, detail="Invalid token audience"
                    )
                except jwt.InvalidSignatureError:
                    print__token_debug("JWT token has invalid signature")
                    raise HTTPException(
                        status_code=401, detail="Invalid token signature"
                    )
                except jwt.DecodeError as e:
                    print__token_debug(f"JWT decode error: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token format")
                except jwt.InvalidTokenError as e:
                    print__token_debug(f"JWT token is invalid: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token")
                except Exception as e:
                    print__token_debug(f"JWT decode error: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token")

        print__token_debug("JWT public key not found in Google JWKS")
        raise HTTPException(
            status_code=401, detail="Invalid token: public key not found"
        )

    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except requests.RequestException as e:
        print__token_debug(f"Failed to fetch Google JWKS: {e}")
        raise HTTPException(
            status_code=401, detail="Token verification failed - unable to validate"
        )
    except jwt.DecodeError as e:
        # This should be rare now due to pre-validation
        print__token_debug(f"JWT decode error in main handler: {e}")
        raise HTTPException(status_code=401, detail="Invalid JWT token format")
    except KeyError as e:
        print__token_debug(f"JWT verification KeyError: {e}")
        raise HTTPException(status_code=401, detail="Invalid JWT token structure")
    except Exception as e:
        print__token_debug(f"JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed")
