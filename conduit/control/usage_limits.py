"""GetUsageLimits — proactive "credits remaining".

OWNER: Lane E. Registered by conduit/app.py; do not edit app.py to wire it.

Confirmed present in Amazon's own client: the AMZ target
`AmazonCodeWhispererService.GetUsageLimits`, with GetUsageLimitsInput (carries
a `resource` field), GetUsageLimitsOutput, and both serializer and
deserializer.  Returns UsageBreakdown{currency, current, overage, usage}.

Why it matters: it flips credit exhaustion from reactive-402 to predictive.
Note there is a DAILY limit (StopReason.DAILY_REQUEST_COUNT) as well as the
monthly one.

REGIONAL HAZARD — do not try to solve this.
conduit/config.py:179-182 carries an upstream note (issue #58) that
codewhisperer.{region}.amazonaws.com does not exist for non-us-east-1 regions,
which is why the runtime host moved to runtime.{region}.kiro.dev. Yet the
control-plane template is still used for ListAvailableProfiles and /mcp. So
GetUsageLimits may only resolve in us-east-1. This module targets the
control-plane host because that is where Amazon's own kiro-cli client places
the operation. A maintainer with a non-us-east-1 profile MUST confirm whether
the call resolves before relying on this endpoint in production.

Sibling operations confirmed in kiro-cli.exe but NOT built here:
  - GetProfile
  - ListAvailableProfiles     (implemented in foundation/auth.py)
  - ListAvailableModels       (model catalog queryable per-account — could
                               replace the static FALLBACK_MODELS list in
                               config.py and power a live /v1/models)
  - CreateSubscriptionToken
  - SendTelemetryEvent

In-repo precedent followed exactly: PROFILE_DISCOVERY_AMZ_TARGET in config.py
and resolve_profile_arn / _request_available_profile_arn in foundation/auth.py
show the working control-plane call shape. get_usage_limits() mirrors it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

from conduit.config import (
    KIRO_CODEWHISPERER_HOST_TEMPLATE,
    PROFILE_ARN,
    PROXY_API_KEY,
    REFRESH_TOKEN,
    REGION,
    USAGE_LIMITS_AMZ_TARGET,
)
from conduit.foundation.auth import KiroAuthManager, resolve_profile_arn
from conduit.foundation.utils import get_kiro_headers


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class UsageBreakdown:
    """Parsed output of GetUsageLimits / usageBreakdown field."""
    currency: str
    current: float
    overage: float
    usage: float


@dataclass
class MeteringUsageInfo:
    """Optional metering detail returned alongside UsageBreakdown."""
    value: float
    unit: str
    unit_plural: str
    form: str
    url: str


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------

_USAGE_CACHE_TTL: int = 60  # seconds — balance check, not per-request

# resource_arn -> (UsageBreakdown, monotonic_timestamp)
_usage_cache: dict[str, tuple[UsageBreakdown, float]] = {}


def _cache_get(resource: str) -> Optional[UsageBreakdown]:
    """Return cached UsageBreakdown if still within TTL, else None."""
    entry = _usage_cache.get(resource)
    if entry is None:
        return None
    result, fetched_at = entry
    if time.monotonic() - fetched_at < _USAGE_CACHE_TTL:
        return result
    _usage_cache.pop(resource, None)
    return None


def _cache_set(resource: str, result: UsageBreakdown) -> None:
    """Store result in cache with a fresh timestamp."""
    _usage_cache[resource] = (result, time.monotonic())


# ---------------------------------------------------------------------------
# Client function
# ---------------------------------------------------------------------------

async def get_usage_limits(
    auth: KiroAuthManager,
    resource: str,
    *,
    _transport: Optional[httpx.AsyncBaseTransport] = None,
) -> UsageBreakdown:
    """Call GetUsageLimits on the CodeWhisperer control-plane host.

    Targets codewhisperer.{region}.amazonaws.com — NOT runtime.{region}.kiro.dev.
    That is where Amazon's own client places this operation.

    REGIONAL HAZARD: the control-plane host may only resolve in us-east-1.
    See module docstring and upstream issue #58.

    Results are cached for _USAGE_CACHE_TTL seconds to avoid per-request hits.

    Args:
        auth:        KiroAuthManager supplying bearer token and api_region.
        resource:    Profile ARN (the ``resource`` field of GetUsageLimitsInput).
        _transport:  Optional httpx transport override — for unit tests only.
                     Production callers must omit it.

    Returns:
        Parsed UsageBreakdown dataclass.

    Raises:
        httpx.HTTPStatusError: On non-2xx upstream responses.
        ValueError:            If the response body lacks expected fields.
    """
    cached = _cache_get(resource)
    if cached is not None:
        logger.debug(f"get_usage_limits: cache hit for resource={resource!r}")
        return cached

    token = await auth.get_access_token()

    # Follow the exact shape of _request_available_profile_arn():
    # 1. Build headers via get_kiro_headers(), then overwrite x-amz-target.
    # 2. POST to the control-plane host with the operation payload.
    headers = get_kiro_headers(auth, token)
    headers["x-amz-target"] = USAGE_LIMITS_AMZ_TARGET

    region = auth.api_region
    url = f"{KIRO_CODEWHISPERER_HOST_TEMPLATE.format(region=region)}/GetUsageLimits"
    payload_bytes = json.dumps({"resource": resource}).encode()

    logger.debug(f"get_usage_limits: POST {url}")

    client_kwargs: dict = {"timeout": 30, "follow_redirects": True}
    if _transport is not None:
        client_kwargs["transport"] = _transport

    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.post(url, headers=headers, content=payload_bytes)
        response.raise_for_status()
        data = response.json()

    # The spec shape is {"usageBreakdown": {...}}; fall back to the root if absent.
    breakdown_raw = data.get("usageBreakdown") or data
    try:
        result = UsageBreakdown(
            currency=breakdown_raw["currency"],
            current=float(breakdown_raw["current"]),
            overage=float(breakdown_raw["overage"]),
            usage=float(breakdown_raw["usage"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"GetUsageLimits response missing expected fields: {exc}"
        ) from exc

    _cache_set(resource, result)
    logger.info(
        f"get_usage_limits: currency={result.currency} "
        f"current={result.current} overage={result.overage} usage={result.usage}"
    )
    return result


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()
_security = HTTPBearer()


def _verify_proxy_key(
    credentials: HTTPAuthorizationCredentials = Security(_security),
) -> str:
    """Dependency: validate the caller's PROXY_API_KEY bearer token."""
    if credentials.credentials != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid proxy API key")
    return credentials.credentials


@router.get("/v1/usage")
async def get_usage_endpoint(
    _key: str = Depends(_verify_proxy_key),
) -> dict:
    """Return remaining credits and usage breakdown.

    Calls GetUsageLimits on the CodeWhisperer control-plane host and returns
    the parsed result.  The response is cached for 60 seconds per resource.

    REGIONAL HAZARD: may only work for profiles in us-east-1.
    See module docstring and upstream issue #58.

    Returns:
        JSON object with keys: resource, currency, current, overage, usage.

    Raises:
        503: No profile ARN available (set PROFILE_ARN in .env).
        502: Upstream GetUsageLimits call failed.
        401: Invalid or missing PROXY_API_KEY.
    """
    auth = KiroAuthManager(
        refresh_token=REFRESH_TOKEN,
        profile_arn=PROFILE_ARN,
        region=REGION,
    )

    try:
        resource = await resolve_profile_arn(auth)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not resolve profile ARN: {exc}"
        ) from exc

    if not resource:
        raise HTTPException(
            status_code=503,
            detail=(
                "No profile ARN available. "
                "Set PROFILE_ARN in .env or ensure credentials support auto-discovery."
            ),
        )

    try:
        breakdown = await get_usage_limits(auth, resource)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"GetUsageLimits upstream error: {exc.response.text[:200]}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "resource": resource,
        "currency": breakdown.currency,
        "current": breakdown.current,
        "overage": breakdown.overage,
        "usage": breakdown.usage,
    }
