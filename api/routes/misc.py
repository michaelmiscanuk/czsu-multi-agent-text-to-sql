# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv
import traceback

load_dotenv()

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
from typing import List
from fastapi import APIRouter, Depends
from fastapi.responses import Response

from api.helpers import traceback_json_response
from api.dependencies.auth import get_current_user

# Create router for miscellaneous endpoints
router = APIRouter()


@router.get("/placeholder/{width}/{height}")
async def get_placeholder_image(width: int, height: int):
    """Generate a placeholder image with specified dimensions. """
    try:
        # Validate dimensions
        width = max(1, min(width, 2000))  # Limit between 1 and 2000 pixels
        height = max(1, min(height, 2000))

        # Create a simple SVG placeholder
        svg_content = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#e5e7eb"/>
            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9ca3af" font-size="20">{width}x{height}</text>
        </svg>"""

        return Response(
            content=svg_content,
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
            },
        )

    except Exception as e:
        resp = traceback_json_response(e)
        if resp is not None:
            # For SVG, return the traceback as text in SVG format
            tb = resp.body.decode() if hasattr(resp, "body") else str(resp)
            tb_svg = f"""<svg width=\"1000\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"100%\" height=\"100%\" fill=\"#f3f4f6\"/><text x=\"10\" y=\"20\" fill=\"#6b7280\" font-size=\"12\">{tb.replace('<','&lt;').replace('>','&gt;')}</text></svg>"""
            return Response(
                content=tb_svg,
                media_type="image/svg+xml",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*",
                },
            )
        simple_svg = f"""<svg width=\"100\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"100%\" height=\"100%\" fill=\"#f3f4f6\"/><text x=\"50%\" y=\"50%\" dominant-baseline=\"middle\" text-anchor=\"middle\" fill=\"#6b7280\" font-size=\"12\">Error</text></svg>"""
        return Response(
            content=simple_svg,
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
            },
        )


@router.get("/initial-followup-prompts", response_model=List[str])
async def get_initial_followup_prompts(_: str = Depends(get_current_user)):
    """
    Generate initial follow-up prompt suggestions for new conversations using AI.

    This endpoint uses AI to generate starter suggestions that will be displayed
    to users when they start a new chat, giving them ideas for questions they can
    ask about Czech Statistical Office data.

    Returns:
        List[str]: A list of AI-generated suggested follow-up prompts for the user
    """
    from main import generate_initial_followup_prompts

    print("🌐 [API] /initial-followup-prompts endpoint called")

    try:
        prompts = generate_initial_followup_prompts()
        return prompts if prompts else []
    except Exception as e:
        print(f"❌ [API] Error generating initial prompts: {str(e)}")
        print(f"❌ [API] Error type: {type(e).__name__}")
        print(f"❌ [API] Traceback: {traceback.format_exc()}")
        return []
