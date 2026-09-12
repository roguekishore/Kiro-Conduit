"""Kiro response-stream event names and the dispatched-event shape.

Event names are verbatim from Amazon's own `kiro-cli.exe` (PLATFORM-FINDINGS
5a-bis). They arrive on the AWS event-stream `:event-type` frame HEADER, which
is what dispatch keys on — never a payload substring match. The upstream
gateway substring-matched 7 patterns that resolved to only 4 of these members.

FROZEN. Lane A parses into these; Lanes C/D render from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Union


class EventName(str, Enum):
    """The ~17 assistant-path events Amazon's own client compiles in."""

    ASSISTANT_RESPONSE = "assistantResponseEvent"
    REASONING_CONTENT = "reasoningContentEvent"
    TOOL_USE = "toolUseEvent"
    TOOL_RESULT = "toolResultEvent"
    METERING = "meteringEvent"
    METADATA = "metadataEvent"
    MESSAGE_METADATA = "messageMetadataEvent"
    CONTEXT_USAGE = "contextUsageEvent"
    CODE = "codeEvent"
    CODE_REFERENCE = "codeReferenceEvent"
    FOLLOWUP_PROMPT = "followupPromptEvent"
    INTENTS = "intentsEvent"
    INVALID_STATE = "invalidStateEvent"
    SUPPLEMENTARY_LINKS = "supplementaryWebLinksEvent"
    CITATION = "citationEvent"
    DOCUMENT_CITATION = "documentCitationEvent"
    MESSAGE_STOP = "messageStopEvent"


@dataclass
class KiroEvent:
    """A recognized event: its header name plus the decoded payload."""

    name: EventName
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnknownEvent:
    """An unrecognized `:event-type`.

    Carried, logged at debug, never silently dropped. The real protocol is
    known to be larger than any list assembled from a strings dump, so an
    unknown name is expected rather than exceptional. Never guess the member
    from payload shape — that is the upstream defect being replaced.
    """

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)


DispatchedEvent = Union[KiroEvent, UnknownEvent]
