"""CONDUIT — a translating gateway between model-native harnesses and Kiro.

Layout:
    config.py      env-driven settings (moved from upstream foundation/)
    contracts/     frozen shared types. Every lane reads; only Lane 0 writes.
    foundation/    auth, account rotation, http, cache, errors. Upstream, kept.
    kiro/          Kiro protocol: response parse (Lane A), request build (Lane B)
    doors/         inbound APIs: Anthropic (Lane C), Codex (Lane D)
    control/       CodeWhisperer control-plane calls (Lane E)
    seam/          fire-and-forget emit to SAGA (Lane F)

See SPEC-WORKSTREAMS.md for lane ownership and the file-collision contract.
"""

__version__ = "0.1.0"
