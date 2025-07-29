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


# ==============================================================================
# DEBUG FUNCTIONS
# ==============================================================================
def print__api_postgresql(msg: str) -> None:
    """Print API-PostgreSQL messages when debug mode is enabled."""
    debug_mode = os.environ.get("print__api_postgresql", "0")
    if debug_mode == "1":
        print(f"[print__api_postgresql] {msg}")
        import sys

        sys.stdout.flush()


def print__feedback_flow(msg: str) -> None:
    """Print FEEDBACK-FLOW messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("DEBUG", "0")
    if debug_mode == "1":
        print(f"[FEEDBACK-FLOW] {msg}")
        import sys

        sys.stdout.flush()


def print__token_debug(msg: str) -> None:
    """Print print__token_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__token_debug", "0")
    if debug_mode == "1":
        print(f"[print__token_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__sentiment_flow(msg: str) -> None:
    """Print SENTIMENT-FLOW messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("DEBUG", "0")
    if debug_mode == "1":
        print(f"[SENTIMENT-FLOW] {msg}")
        import sys

        sys.stdout.flush()


def print__debug(msg: str) -> None:
    """Print DEBUG messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("DEBUG", "0")
    if debug_mode == "1":
        print(f"[DEBUG] {msg}")
        import sys

        sys.stdout.flush()


def print__memory_debug(msg: str) -> None:
    """Print DEBUG messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__memory_debug", "0")
    if print__memory_debug == "1":
        print(f"[print__memory_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__main_debug(msg: str) -> None:
    """Print DEBUG messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__main_debug", "0")
    if print__main_debug == "1":
        print(f"[print__main_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__analyze_debug(msg: str) -> None:
    """Print print__analyze_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__analyze_debug", "0")
    if debug_mode == "1":
        print(f"[print__analyze_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__chat_all_messages_debug(msg: str) -> None:
    """Print print__chat_all_messages_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__chat_all_messages_debug", "0")
    if debug_mode == "1":
        print(f"[print__chat_all_messages_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__chat_all_messages_one_thread_debug(msg: str) -> None:
    """Print print__chat_all_messages_one_thread_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__chat_all_messages_one_thread_debug", "0")
    if debug_mode == "1":
        print(f"[print__chat_all_messages_one_thread_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__feedback_debug(msg: str) -> None:
    """Print print__feedback_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__feedback_debug", "0")
    if debug_mode == "1":
        print(f"[print__feedback_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__sentiment_debug(msg: str) -> None:
    """Print print__sentiment_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__sentiment_debug", "0")
    if debug_mode == "1":
        print(f"[print__sentiment_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__chat_threads_debug(msg: str) -> None:
    """Print print__chat_threads_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__chat_threads_debug", "0")
    if debug_mode == "1":
        print(f"[print__chat_threads_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__chat_messages_debug(msg: str) -> None:
    """Print print__chat_messages_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__chat_messages_debug", "0")
    if debug_mode == "1":
        print(f"[print__chat_messages_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__delete_chat_debug(msg: str) -> None:
    """Print print__delete_chat_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__delete_chat_debug", "0")
    if debug_mode == "1":
        print(f"[print__delete_chat_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__chat_sentiments_debug(msg: str) -> None:
    """Print print__chat_sentiments_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__chat_sentiments_debug", "0")
    if debug_mode == "1":
        print(f"[print__chat_sentiments_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__catalog_debug(msg: str) -> None:
    """Print print__catalog_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__catalog_debug", "0")
    if debug_mode == "1":
        print(f"[print__catalog_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__data_tables_debug(msg: str) -> None:
    """Print print__data_tables_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__data_tables_debug", "0")
    if debug_mode == "1":
        print(f"[print__data_tables_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__data_table_debug(msg: str) -> None:
    """Print print__data_table_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__data_table_debug", "0")
    if debug_mode == "1":
        print(f"[print__data_table_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__chat_thread_id_checkpoints_debug(msg: str) -> None:
    """Print print__chat_thread_id_checkpoints_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__chat_thread_id_checkpoints_debug", "0")
    if debug_mode == "1":
        print(f"[print__chat_thread_id_checkpoints_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__debug_pool_status_debug(msg: str) -> None:
    """Print print__debug_pool_status_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__debug_pool_status_debug", "0")
    if debug_mode == "1":
        print(f"[print__debug_pool_status_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__chat_thread_id_run_ids_debug(msg: str) -> None:
    """Print print__chat_thread_id_run_ids_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__chat_thread_id_run_ids_debug", "0")
    if debug_mode == "1":
        print(f"[print__chat_thread_id_run_ids_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__debug_run_id_debug(msg: str) -> None:
    """Print print__debug_run_id_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__debug_run_id_debug", "0")
    if debug_mode == "1":
        print(f"[print__debug_run_id_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__admin_clear_cache_debug(msg: str) -> None:
    """Print admin clear cache debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    admin_clear_cache_debug_mode = os.environ.get("ADMIN_CLEAR_CACHE_DEBUG", "0")
    if admin_clear_cache_debug_mode == "1":
        print(f"[ADMIN_CLEAR_CACHE_DEBUG] {msg}")
        import sys

        sys.stdout.flush()


def print__analysis_tracing_debug(msg: str) -> None:
    """Print analysis tracing debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    analysis_tracing_debug_mode = os.environ.get("print__analysis_tracing_debug", "0")
    if analysis_tracing_debug_mode == "1":
        print(f"[print__analysis_tracing_debug] 🔍 {msg}")
        import sys

        sys.stdout.flush()


def print__startup_debug(msg: str) -> None:
    """Print startup debug messages when debug mode is enabled."""
    debug_mode = os.environ.get("DEBUG", "0")
    if debug_mode == "1":
        print(f"[STARTUP-DEBUG] {msg}")
        sys.stdout.flush()


def print__memory_monitoring(msg: str) -> None:
    """Print MEMORY-MONITORING messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("DEBUG", "0")
    if debug_mode == "1":
        print(f"[MEMORY-MONITORING] {msg}")
        sys.stdout.flush()


def print__nodes_debug(msg: str) -> None:
    """Print print__nodes_debug messages when debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__nodes_debug", "0")
    if debug_mode == "1":
        print(f"[print__nodes_debug] {msg}")
        import sys

        sys.stdout.flush()


def print__tools_debug(msg: str) -> None:
    """Print TOOLS DEBUG messages when tools debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__tools_debug", "0")
    if debug_mode == "1":
        print(f"[TOOLS] {msg}")
        import sys

        sys.stdout.flush()


def print__checkpointers_debug(msg: str) -> None:
    """Print print__checkpointers_debug messages when tools debug mode is enabled.

    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get("print__checkpointers_debug", "0")
    if debug_mode == "1":
        print(f"[print__checkpointers_debug] {msg}")
        import sys

        sys.stdout.flush()
