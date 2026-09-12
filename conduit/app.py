"""FastAPI application factory and the single router-registration point.

OWNER: Lane 0. This file is FROZEN for Lanes A-F.

Every door and control router is imported and registered here, once. Lane C,
D, and E fill in the `router` object inside their own module — they never edit
this file. That is what removes the only real file-collision point between
otherwise-parallel lanes.

Endpoints owned here:
    GET /health     liveness. Unauthenticated by design.
    GET /v1/models  the model catalog, from config.FALLBACK_MODELS.

AUTH NOTE: /health and /v1/models are intentionally unauthenticated — they
expose no account data and no credentials. Every door route MUST authenticate
via PROXY_API_KEY; that is each door lane's responsibility, enforced in the
door module, not here. Do not add a route to this file that returns account,
usage, or credential data without an auth dependency.

Bind host/port come from config.SERVER_HOST / SERVER_PORT. Default is
loopback; do not widen it without a maintainer's decision.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger

from conduit import __version__
from conduit.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    FALLBACK_MODELS,
    HIDDEN_FROM_LIST,
)
from conduit.control import usage_limits
from conduit.doors import anthropic, codex
from conduit.foundation.exceptions import validation_exception_handler


def create_app() -> FastAPI:
    """Build the CONDUIT application with every router registered."""
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=__version__,
    )

    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Doors (inbound harness APIs) and the control plane.
    # Each router is owned by its lane; registration order is not significant.
    app.include_router(anthropic.router, tags=["anthropic"])
    app.include_router(codex.router, tags=["codex"])
    app.include_router(usage_limits.router, tags=["control"])

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models", tags=["meta"])
    async def list_models() -> dict:
        """Model catalog.

        Sourced from config.FALLBACK_MODELS. Note that Kiro also exposes
        ListAvailableModels on the control plane, so the catalog is queryable
        per-account rather than necessarily hardcoded — see Lane E's docstring.
        """
        models = [
            {"id": m["modelId"], "object": "model", "owned_by": "kiro"}
            for m in FALLBACK_MODELS
            if m.get("modelId") not in HIDDEN_FROM_LIST
        ]
        return {"object": "list", "data": models}

    logger.debug(f"CONDUIT app created, {len(app.routes)} routes registered")
    return app
