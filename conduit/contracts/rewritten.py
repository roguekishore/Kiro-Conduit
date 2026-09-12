"""The final Kiro-shaped request, plus CONDUIT's self-declared injections.

SAGA captures the clean-in request at its front door but cannot see the
rewrite, because the rewrite happens inside CONDUIT. This is channel 2 of the
seam.

CONDUIT declares what it changed rather than making SAGA reverse-engineer the
clean-in diff. SAGA may still diff to verify.

FROZEN. Lane B populates; Lane F serializes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class InjectionType(str, Enum):
    THINKING_TAGS = "thinking_tags"
    SYSTEM_PROMPT_SLOT = "system_prompt_slot"
    TURN_MERGE = "turn_merge"
    RELOCATED_TOOL_DOCS = "relocated_tool_docs"


@dataclass
class Injection:
    type: InjectionType
    location: str    # e.g. "systemPrompt", "history", "last_user_message"


@dataclass
class RewrittenOut:
    system_prompt: Optional[str] = None
    current_message: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    injections: List[Injection] = field(default_factory=list)
