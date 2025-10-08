import os
import traceback

from fastapi.responses import JSONResponse


def traceback_json_response(e, status_code=500, run_id=None):
    if os.environ.get("DEBUG_TRACEBACK") == "1":
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        response_content = {
            "detail": str(e),
            "traceback": tb_str,
        }
        # Include run_id if provided to maintain consistency with successful responses
        if run_id:
            response_content["run_id"] = run_id
        return JSONResponse(
            status_code=status_code,
            content=response_content,
        )
    return None  # Caller should handle fallback
