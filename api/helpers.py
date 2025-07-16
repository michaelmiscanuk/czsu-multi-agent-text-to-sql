import os
import traceback

from fastapi.responses import JSONResponse


def traceback_json_response(e, status_code=500):
    if os.environ.get("DEBUG_TRACEBACK") == "1":
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return JSONResponse(
            status_code=status_code,
            content={
                "detail": str(e),
                "traceback": tb_str,
            },
        )
    return None  # Caller should handle fallback
