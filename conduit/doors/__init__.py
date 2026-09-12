"""Inbound APIs.

Lane C owns anthropic.py + anthropic_sse.py (/v1/messages, Claude Code).
Lane D owns codex.py + codex_dialect.py + codex_sse.py (/v1/responses, Codex CLI).

Routers are registered once by conduit/app.py, which is Lane 0's file. Door
lanes fill in the router in their own module and never edit app.py.
"""
