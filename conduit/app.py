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

SECURITY — BIND ADDRESS AND PROXY KEY ARE BOTH INSECURE BY DEFAULT.
An earlier version of this docstring claimed "Default is loopback"; that was
wrong. The real defaults, both inherited from the retained upstream config, are:

    config.SERVER_HOST   = "0.0.0.0"                     — every interface
    config.PROXY_API_KEY = "my-super-secret-password-123" — a published constant

Together those expose a gateway that holds cloud credentials and fronts a
metered backend to the whole local network, behind a password anyone reading
this repo knows. Neither default is changed here, because narrowing a
network-facing default is a maintainer's decision, not a mechanical fix.
Set SERVER_HOST=127.0.0.1 and a real PROXY_API_KEY in .env before serving.

app.state contract (set by the lifespan handler below, read by every door):
    app.state.account_manager  AccountManager, credentials + state loaded
    app.state.account_system   bool, mirrors config.ACCOUNT_SYSTEM
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger

from conduit import __version__
from conduit.config import (
    ACCOUNT_SYSTEM,
    ACCOUNTS_CONFIG_FILE,
    ACCOUNTS_STATE_FILE,
    APP_DESCRIPTION,
    APP_TITLE,
    FALLBACK_MODELS,
    HIDDEN_FROM_LIST,
)
from conduit.control import usage_limits
from conduit.doors import anthropic, codex
from conduit.foundation.account_manager import AccountManager
from conduit.foundation.exceptions import validation_exception_handler


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Populate app.state before the first request is served.

    Both doors read ``request.app.state.account_manager``. Nothing used to set
    it, so every /v1/messages and /v1/responses request raised AttributeError
    before it could reach Kiro — the doors were unreachable by construction.

    Failures are deliberately NOT swallowed here. ``load_credentials`` and
    ``load_state`` already handle a missing or malformed file internally (they
    log and continue with zero accounts), so anything escaping them is a real
    fault, and a gateway that cannot resolve an account should fail at startup
    rather than turn every request into a 500.

    Note on modes: ``get_next_account`` (ACCOUNT_SYSTEM=true) initializes an
    account lazily on first use, but ``get_first_account`` (legacy mode, and
    ACCOUNT_SYSTEM defaults to false) does not — it raises RuntimeError until
    something has initialized one. Each door converts that into a 503 with a
    clear message; see the account-resolution block there.
    """
    manager = AccountManager(ACCOUNTS_CONFIG_FILE, ACCOUNTS_STATE_FILE)
    await manager.load_credentials()
    await manager.load_state()

    app.state.account_manager = manager
    app.state.account_system = ACCOUNT_SYSTEM
    logger.info(
        f"CONDUIT startup: account_system={ACCOUNT_SYSTEM} "
        f"credentials={ACCOUNTS_CONFIG_FILE!r}"
    )

    yield


def create_app() -> FastAPI:
    """Build the CONDUIT application with every router registered."""
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=__version__,
        lifespan=_lifespan,
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

    @app.get("/api/hello", tags=["meta"])
    async def hello() -> dict:
        """Reachability probe.

        The Kiro CLI probes this path to decide whether a gateway is live.
        Without it the request 404'd, and each failed probe still minted a
        session downstream — filling SAGA with empty sessions that never
        carried a turn. Same unauthenticated shape as /health, so it exposes
        nothing that endpoint did not already.
        """
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
