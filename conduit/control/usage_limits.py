"""GetUsageLimits — proactive "credits remaining".

OWNER: Lane E. Registered by conduit/app.py; do not edit app.py to wire it.

Confirmed present in Amazon's own client: the AMZ target
`AmazonCodeWhispererService.GetUsageLimits`, with GetUsageLimitsInput (carries
a `resource` field), GetUsageLimitsOutput, and both serializer and
deserializer. Returns UsageBreakdown{currency, current, overage, usage}.

Why it matters: it flips credit exhaustion from reactive-402 to predictive.
Note there is a DAILY limit (StopReason.DAILY_REQUEST_COUNT) as well as the
monthly one.

REGIONAL HAZARD — flag it, do not try to solve it. conduit/config.py:179-182
carries an upstream note (issue #58) that codewhisperer.{region}.amazonaws.com
does not exist for non-us-east-1 regions, which is why the runtime host moved
to runtime.{region}.kiro.dev. Yet the control-plane template is still used for
ListAvailableProfiles and /mcp. So GetUsageLimits may only resolve in
us-east-1. Implement against the control-plane host — that is where Amazon's
own client puts the op — and report the constraint.

In-repo precedent is exact: conduit/config.py PROFILE_DISCOVERY_AMZ_TARGET
plus conduit/foundation/auth.py resolve_profile_arn() is a working
control-plane call. Follow its shape.

See SPEC-WORKSTREAMS.md, Lane E, for the full brief and verification gate.
"""

from fastapi import APIRouter

router = APIRouter()
