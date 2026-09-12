"""Lane E verification gate — tests for conduit/control/usage_limits.py.

All tests run against a mocked httpx transport.  No live network calls are made.

Assertions:
- Request targets the control-plane host (codewhisperer.{region}.amazonaws.com),
  NOT the runtime host (runtime.{region}.kiro.dev).
- x-amz-target header is exactly USAGE_LIMITS_AMZ_TARGET.
- A fixture response parses into UsageBreakdown with correct field values.
- TTL caching suppresses a second HTTP call within the TTL window.
- Cache miss after TTL expiry triggers a fresh HTTP call.
- The /v1/usage endpoint returns 200 with a valid JSON body when mocked.
- The /v1/usage endpoint returns 401 for a missing / wrong API key.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from conduit.config import USAGE_LIMITS_AMZ_TARGET
from conduit.control import usage_limits
from conduit.control.usage_limits import (
    UsageBreakdown,
    _usage_cache,
    get_usage_limits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockTransport(httpx.AsyncBaseTransport):
    """Intercepts every async request and returns a pre-configured JSON response.

    Exposes `requests` so tests can inspect the calls that were made.
    """

    def __init__(self, response_body: dict[str, Any], status_code: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self._body = response_body
        self._status_code = status_code

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            status_code=self._status_code,
            headers={"content-type": "application/json"},
            content=json.dumps(self._body).encode(),
        )


def _make_auth(region: str = "us-east-1") -> Any:
    """Return a minimal auth mock accepted by get_kiro_headers()."""
    from unittest.mock import MagicMock
    auth = MagicMock()
    auth.get_access_token = AsyncMock(return_value="fake-access-token")
    auth.api_region = region
    auth.fingerprint = "deadbeefdeadbeef" * 4  # 64 hex chars
    return auth


SAMPLE_RESOURCE = "arn:aws:codewhisperer:us-east-1:123456789012:profile/test-profile"

SAMPLE_RESPONSE = {
    "usageBreakdown": {
        "currency": "USD",
        "current": 100.0,
        "overage": 5.0,
        "usage": 42.5,
    }
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Wipe the module-level usage cache before every test."""
    _usage_cache.clear()
    yield
    _usage_cache.clear()


@pytest.fixture
def auth():
    return _make_auth()


@pytest.fixture
def transport():
    return _MockTransport(SAMPLE_RESPONSE)


# ---------------------------------------------------------------------------
# 1. Request targets the control-plane host, NOT the runtime host
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_targets_control_plane_host(auth, transport):
    await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=transport)

    assert len(transport.requests) == 1, "Expected exactly one HTTP request"
    url = str(transport.requests[0].url)

    assert "codewhisperer.us-east-1.amazonaws.com" in url, (
        f"Expected control-plane host in URL, got: {url}"
    )
    assert "runtime" not in url, (
        f"URL must NOT target the runtime host, got: {url}"
    )
    assert "kiro.dev" not in url, (
        f"URL must NOT target kiro.dev, got: {url}"
    )


@pytest.mark.asyncio
async def test_request_url_contains_get_usage_limits_path(auth, transport):
    await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=transport)

    url = str(transport.requests[0].url)
    assert url.endswith("/GetUsageLimits"), f"Expected /GetUsageLimits path suffix, got: {url}"


# ---------------------------------------------------------------------------
# 2. x-amz-target header is exactly USAGE_LIMITS_AMZ_TARGET
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_amz_target_header_value(auth, transport):
    await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=transport)

    req = transport.requests[0]
    actual_target = req.headers.get("x-amz-target")
    assert actual_target == USAGE_LIMITS_AMZ_TARGET, (
        f"x-amz-target must be {USAGE_LIMITS_AMZ_TARGET!r}, got {actual_target!r}"
    )


@pytest.mark.asyncio
async def test_amz_target_constant_value():
    """The constant itself must carry the correct string — catches typos."""
    assert USAGE_LIMITS_AMZ_TARGET == "AmazonCodeWhispererService.GetUsageLimits"


# ---------------------------------------------------------------------------
# 3. Fixture response parses into UsageBreakdown with correct field values
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_response_parses_to_usage_breakdown(auth, transport):
    result = await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=transport)

    assert isinstance(result, UsageBreakdown)
    assert result.currency == "USD"
    assert result.current == 100.0
    assert result.overage == 5.0
    assert result.usage == 42.5


@pytest.mark.asyncio
async def test_response_parses_flat_body(auth):
    """UsageBreakdown fields may be at root level (no usageBreakdown wrapper)."""
    flat_response = {"currency": "EUR", "current": 200.0, "overage": 0.0, "usage": 10.0}
    t = _MockTransport(flat_response)

    result = await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=t)

    assert isinstance(result, UsageBreakdown)
    assert result.currency == "EUR"
    assert result.current == 200.0


@pytest.mark.asyncio
async def test_missing_required_field_raises_value_error(auth):
    """A response body missing 'currency' should raise ValueError."""
    bad_response = {"usageBreakdown": {"current": 1.0, "overage": 0.0, "usage": 0.5}}
    t = _MockTransport(bad_response)

    with pytest.raises(ValueError, match="missing expected fields"):
        await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=t)


# ---------------------------------------------------------------------------
# 4. TTL caching suppresses a second HTTP call within the TTL window
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ttl_cache_suppresses_second_call(auth, transport):
    # First call should reach the network.
    result1 = await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=transport)
    assert len(transport.requests) == 1

    # Second call within TTL must NOT produce a new HTTP request.
    result2 = await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=transport)
    assert len(transport.requests) == 1, (
        "Second call within TTL window must be served from cache (no new HTTP request)"
    )

    assert result1 == result2


@pytest.mark.asyncio
async def test_different_resources_each_make_a_request(auth, transport):
    """Cache is keyed on resource; two distinct resources must each hit the network."""
    res_a = "arn:aws:codewhisperer:us-east-1:111:profile/a"
    res_b = "arn:aws:codewhisperer:us-east-1:111:profile/b"

    await get_usage_limits(auth, res_a, _transport=transport)
    await get_usage_limits(auth, res_b, _transport=transport)

    assert len(transport.requests) == 2, (
        "Each distinct resource should produce its own HTTP request"
    )


@pytest.mark.asyncio
async def test_cache_miss_after_ttl_expiry(auth, transport):
    """After the TTL window has elapsed, a fresh request must be made."""
    # Populate the cache.
    await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=transport)
    assert len(transport.requests) == 1

    # Wind the cache entry's timestamp back past the TTL.
    breakdown, _ = _usage_cache[SAMPLE_RESOURCE]
    _usage_cache[SAMPLE_RESOURCE] = (breakdown, time.monotonic() - 9999)

    # Now a fresh call should hit the network again.
    await get_usage_limits(auth, SAMPLE_RESOURCE, _transport=transport)
    assert len(transport.requests) == 2, (
        "Stale cache entry must not suppress a new HTTP request"
    )


# ---------------------------------------------------------------------------
# 5. FastAPI router — /v1/usage endpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def test_app():
    """Minimal FastAPI app wiring only the Lane E router."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(usage_limits.router)
    return app


@pytest.mark.asyncio
async def test_usage_endpoint_returns_200_with_mocked_internals(test_app):
    """GET /v1/usage returns 200 and a correctly shaped body when internals are mocked."""
    fake_breakdown = UsageBreakdown(
        currency="USD", current=100.0, overage=0.0, usage=50.0
    )
    fake_arn = SAMPLE_RESOURCE

    with (
        patch(
            "conduit.control.usage_limits.KiroAuthManager",
            return_value=_make_auth(),
        ),
        patch(
            "conduit.control.usage_limits.resolve_profile_arn",
            new=AsyncMock(return_value=fake_arn),
        ),
        patch(
            "conduit.control.usage_limits.get_usage_limits",
            new=AsyncMock(return_value=fake_breakdown),
        ),
    ):
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/usage",
                headers={"Authorization": "Bearer my-super-secret-password-123"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["currency"] == "USD"
    assert body["current"] == 100.0
    assert body["usage"] == 50.0
    assert body["resource"] == fake_arn


@pytest.mark.asyncio
async def test_usage_endpoint_returns_401_without_auth(test_app):
    """GET /v1/usage without an Authorization header must be rejected (401 or 403)."""
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get("/v1/usage")

    # FastAPI HTTPBearer rejects missing credentials; the exact code (401/403)
    # varies by FastAPI version — either is a valid rejection.
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_usage_endpoint_returns_401_with_wrong_key(test_app):
    """GET /v1/usage with the wrong API key must return 401."""
    with (
        patch("conduit.control.usage_limits.KiroAuthManager", return_value=_make_auth()),
        patch(
            "conduit.control.usage_limits.resolve_profile_arn",
            new=AsyncMock(return_value=SAMPLE_RESOURCE),
        ),
        patch(
            "conduit.control.usage_limits.get_usage_limits",
            new=AsyncMock(return_value=UsageBreakdown("USD", 0.0, 0.0, 0.0)),
        ),
    ):
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/usage",
                headers={"Authorization": "Bearer wrong-key"},
            )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_usage_endpoint_returns_503_when_no_profile_arn(test_app):
    """GET /v1/usage returns 503 when profile ARN cannot be resolved."""
    with (
        patch("conduit.control.usage_limits.KiroAuthManager", return_value=_make_auth()),
        patch(
            "conduit.control.usage_limits.resolve_profile_arn",
            new=AsyncMock(return_value=None),
        ),
    ):
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/usage",
                headers={"Authorization": "Bearer my-super-secret-password-123"},
            )

    assert response.status_code == 503
