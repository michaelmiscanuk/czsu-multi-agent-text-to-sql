"""Miscellaneous Utility Routes for CZSU Multi-Agent Text-to-SQL API

This module provides utility endpoints for the CZSU multi-agent text-to-SQL system,
including placeholder image generation and AI-powered followup prompt suggestions
for new conversations.
"""

MODULE_DESCRIPTION = r"""Miscellaneous Utility Routes for CZSU Multi-Agent Text-to-SQL API

This module serves as a collection of utility endpoints that support the core functionality
of the CZSU multi-agent text-to-SQL system. It provides helper services for the frontend
interface, including dynamic placeholder image generation and AI-powered conversation starters.

Key Features:
-------------
1. Placeholder Image Generation:
   - Dynamic SVG image creation with custom dimensions
   - Dimension validation and safety limits (1-2000 pixels)
   - Lightweight vector graphics for responsive design
   - Dimension text overlay for development/debugging
   - Caching headers for browser performance optimization
   - CORS headers for cross-origin compatibility
   - Graceful error handling with error-state SVG responses

2. AI-Powered Followup Prompts:
   - Initial conversation starter suggestions using AI
   - Helps users begin conversations with meaningful queries
   - Provides contextual examples of CZSU data queries
   - Integration with main application's prompt generation logic
   - Empty array fallback for graceful error handling
   - Detailed error logging for troubleshooting

API Endpoints:
-------------
1. GET /placeholder/{width}/{height}
   - Generate placeholder SVG images with specified dimensions
   - Path parameters: width (int), height (int)
   - Returns: SVG image with dimension overlay
   - Use case: Frontend loading states, layout testing

2. GET /initial-followup-prompts
   - Get AI-generated conversation starter prompts
   - Returns: Array of suggested prompt strings
   - Authentication: Required (JWT token)
   - Use case: New conversation initialization"""

# ==============================================================================
# CRITICAL: WINDOWS EVENT LOOP POLICY CONFIGURATION
# ==============================================================================

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
# with Windows systems. The WindowsSelectorEventLoopPolicy resolves issues
# with the default ProactorEventLoop that can cause database connection problems.
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================

# Load environment variables early to ensure configuration availability
from dotenv import load_dotenv
import traceback

load_dotenv()

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Determine base directory for the project, handling both regular execution
# and special cases where __file__ might not be defined (e.g., REPL)
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]  # Navigate up to project root
except NameError:
    # Fallback for environments where __file__ is not available
    BASE_DIR = Path(os.getcwd()).parents[0]

# ==============================================================================
# STANDARD LIBRARY IMPORTS
# ==============================================================================

# Type hints for improved code clarity and IDE support
from typing import List

# ==============================================================================
# THIRD-PARTY IMPORTS - WEB FRAMEWORK
# ==============================================================================

# FastAPI components for building REST API endpoints
from fastapi import APIRouter, Depends
from fastapi.responses import Response

# ==============================================================================
# API DEPENDENCIES AND HELPERS
# ==============================================================================

# Custom error response formatting for client debugging
from api.helpers import traceback_json_response

# JWT-based authentication dependency for protecting endpoints
from api.dependencies.auth import get_current_user

# ==============================================================================
# FASTAPI ROUTER INITIALIZATION
# ==============================================================================

# Create router instance for miscellaneous utility endpoints
router = APIRouter()


# ==============================================================================
# API ENDPOINT: PLACEHOLDER IMAGE GENERATION
# ==============================================================================


@router.get("/placeholder/{width}/{height}")
async def get_placeholder_image(width: int, height: int):
    """Generate a placeholder SVG image with specified dimensions.

    This endpoint creates a lightweight SVG image with custom width and height,
    displaying the dimensions as centered text. Useful for frontend development,
    layout testing, and loading states.

    Key Features:
        - Dynamic SVG generation with custom dimensions
        - Automatic dimension validation and safety clamping
        - Lightweight vector graphics (sub-KB file sizes)
        - Browser caching for performance optimization
        - CORS-enabled for cross-origin usage
        - Graceful error handling with error-state SVGs

    Dimension Constraints:
        - Minimum: 1x1 pixels
        - Maximum: 2000x2000 pixels
        - Out-of-range values automatically clamped to valid range
        - Prevents resource exhaustion from excessive dimensions

    Args:
        width: Desired image width in pixels (clamped to 1-2000)
        height: Desired image height in pixels (clamped to 1-2000)

    Returns:
        Response: SVG image with dimension text overlay
            - Content-Type: image/svg+xml
            - Cache-Control: public, max-age=3600 (1 hour cache)
            - Access-Control-Allow-Origin: * (CORS enabled)

    Error Handling:
        - On exception: Returns error SVG with traceback details
        - Fallback: Returns simple "Error" SVG if formatting fails
        - Errors never prevent response (always returns valid SVG)

    Example:
        GET /placeholder/800/600
        Returns: <svg width="800" height="600">
                   <rect fill="#e5e7eb"/>
                   <text>800x600</text>
                 </svg>
    """
    try:
        # =======================================================================
        # DIMENSION VALIDATION AND SAFETY CLAMPING
        # =======================================================================

        # Clamp dimensions to safe range (1-2000 pixels)
        # This prevents resource exhaustion from excessive dimensions
        # and ensures reasonable image sizes for typical use cases
        width = max(1, min(width, 2000))  # Minimum 1px, maximum 2000px
        height = max(1, min(height, 2000))

        # =======================================================================
        # SVG GENERATION
        # =======================================================================

        # Create a simple SVG placeholder with centered dimension text
        # Uses inline styles for cross-browser compatibility
        # Light gray background (#e5e7eb) with medium gray text (#9ca3af)
        svg_content = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#e5e7eb"/>
            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9ca3af" font-size="20">{width}x{height}</text>
        </svg>"""

        # =======================================================================
        # RESPONSE WITH CACHING AND CORS HEADERS
        # =======================================================================

        # Return SVG with appropriate headers for performance and compatibility
        return Response(
            content=svg_content,
            media_type="image/svg+xml",  # Proper MIME type for SVG
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Access-Control-Allow-Origin": "*",  # Enable CORS
            },
        )

    except Exception as e:
        # =======================================================================
        # ERROR HANDLING WITH SVG ERROR RESPONSES
        # =======================================================================

        # Try to generate detailed error SVG with traceback
        resp = traceback_json_response(e)
        if resp is not None:
            # Extract traceback text from response
            tb = resp.body.decode() if hasattr(resp, "body") else str(resp)

            # Create error SVG with traceback text
            # Escape HTML entities to prevent XSS and rendering issues
            tb_svg = f"""<svg width=\"1000\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"100%\" height=\"100%\" fill=\"#f3f4f6\"/><text x=\"10\" y=\"20\" fill=\"#6b7280\" font-size=\"12\">{tb.replace('<','&lt;').replace('>','&gt;')}</text></svg>"""
            return Response(
                content=tb_svg,
                media_type="image/svg+xml",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*",
                },
            )

        # Fallback: simple error SVG if detailed formatting fails
        simple_svg = f"""<svg width=\"100\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"100%\" height=\"100%\" fill=\"#f3f4f6\"/><text x=\"50%\" y=\"50%\" dominant-baseline=\"middle\" text-anchor=\"middle\" fill=\"#6b7280\" font-size=\"12\">Error</text></svg>"""
        return Response(
            content=simple_svg,
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
            },
        )


# ==============================================================================
# API ENDPOINT: AI-GENERATED FOLLOWUP PROMPTS
# ==============================================================================


@router.get("/initial-followup-prompts", response_model=List[str])
async def get_initial_followup_prompts(_: str = Depends(get_current_user)):
    """Generate initial follow-up prompt suggestions for new conversations using AI.

    This endpoint uses AI to generate starter suggestions that will be displayed
    to users when they start a new chat, giving them ideas for questions they can
    ask about Czech Statistical Office (CZSU) data.

    The prompts are designed to:
    - Showcase the system's capabilities
    - Provide concrete examples of valid queries
    - Guide users toward meaningful statistical questions
    - Cover diverse topics within CZSU data catalog

    Authentication:
        - Requires valid JWT token via get_current_user dependency
        - Token parameter named "_" as it's used only for auth validation

    Returns:
        List[str]: A list of AI-generated suggested follow-up prompts for the user.
                   Empty list [] if generation fails.

    Example Response:
        [
            "What was the population of Prague in 2020?",
            "Show me unemployment statistics for the last 5 years",
            "Compare GDP growth between Czech regions"
        ]

    Error Handling:
        - Returns empty array [] on any error
        - Errors logged with full details for debugging
        - Never raises exceptions (graceful degradation)

    Note:
        - Imports from main module at runtime to avoid circular dependencies
        - Generation may take 1-2 seconds (AI processing time)
        - Prompts are not cached (generated fresh per request)
    """

    # =======================================================================
    # DYNAMIC IMPORT TO AVOID CIRCULAR DEPENDENCIES
    # =======================================================================

    # Import at function level to prevent circular import issues
    # The main module may import from this routes module, so we delay
    # the import until it's actually needed at runtime
    from main import generate_initial_followup_prompts

    # Log request for monitoring and debugging
    print("🌐 [API] /initial-followup-prompts endpoint called")

    try:
        # =======================================================================
        # AI PROMPT GENERATION
        # =======================================================================

        # Call AI-powered prompt generation function from main module
        # This typically uses an LLM to generate contextual suggestions
        prompts = generate_initial_followup_prompts()

        # =======================================================================
        # RESPONSE VALIDATION AND RETURN
        # =======================================================================

        # Return generated prompts, or empty list if None
        # Ensures we always return a valid list (never None)
        return prompts if prompts else []

    except Exception as e:
        # =======================================================================
        # ERROR HANDLING WITH DETAILED LOGGING
        # =======================================================================

        # Log detailed error information for debugging
        # Uses emoji markers for easy visual parsing in logs
        print(f"❌ [API] Error generating initial prompts: {str(e)}")
        print(f"❌ [API] Error type: {type(e).__name__}")
        print(f"❌ [API] Traceback: {traceback.format_exc()}")

        # Return empty array for graceful degradation
        # Frontend can handle empty list without breaking
        return []
