"""Utilities for streaming LangGraph output to external consumers.

This module provides a lightweight context-based mechanism for sharing
streaming callbacks between the FastAPI layer (which owns the HTTP connection)
and LangGraph nodes (which want to emit incremental updates such as answer tokens).

Usage pattern:
    1. The API sets a coroutine callback via ``set_streaming_callback`` before
       invoking ``analysis_main``.
    2. The LangGraph node (e.g., ``format_answer_node``) retrieves the callback
       implicitly via ``emit_stream_event`` / ``emit_answer_chunk`` and awaits it
       whenever a new piece of content should be streamed to the client.
    3. After the graph finishes, the API resets the context token to avoid
       leaking callbacks across requests.

The callback signature is intentionally simple: it receives a ``dict`` payload
with at minimum a ``type`` key describing the event. This keeps the transport
agnostic (SSE, WebSocket, etc.) while allowing callers to add custom metadata.
"""

from __future__ import annotations

import contextvars
from typing import Any, Awaitable, Callable, Dict, Optional

StreamEvent = Dict[str, Any]
StreamCallback = Callable[[StreamEvent], Awaitable[None]]

_stream_callback_var: contextvars.ContextVar[Optional[StreamCallback]] = (
    contextvars.ContextVar("streaming_callback", default=None)
)


def set_streaming_callback(callback: Optional[StreamCallback]):
    """Register a coroutine callback for streaming events.

    Returns the context token so callers can reset the callback after the
    analysis completes.
    """

    return _stream_callback_var.set(callback)


def reset_streaming_callback(token) -> None:
    """Reset the streaming callback context using the provided token."""

    _stream_callback_var.reset(token)


def get_streaming_callback() -> Optional[StreamCallback]:
    """Return the currently registered streaming callback, if any."""

    return _stream_callback_var.get()


async def emit_stream_event(event: StreamEvent) -> None:
    """Emit a structured streaming event if a callback is registered."""

    callback = _stream_callback_var.get()
    if callback is None:
        return
    if "type" not in event:
        raise ValueError("Streaming events must include a 'type' key")
    await callback(event)


async def emit_answer_chunk(text: str) -> None:
    """Helper for emitting incremental final-answer content chunks."""

    if not text:
        return
    await emit_stream_event({"type": "answer_chunk", "token": text})
