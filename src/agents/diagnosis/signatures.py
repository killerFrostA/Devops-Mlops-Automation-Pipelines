"""Known failure signatures recognized in log, Kubernetes and change evidence."""

from __future__ import annotations

import re

from src.agents.diagnosis.contracts import EvidenceItem


SIGNATURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "oom": re.compile(r"\boomkilled\b|out[- ]of[- ]memory|\boom\b", re.IGNORECASE),
    "crash_loop": re.compile(
        r"crashloopbackoff|back-off restarting failed container", re.IGNORECASE
    ),
    "config_invalid": re.compile(
        r"invalid (?:configuration|config)|configuration error|"
        r"config(?:uration)? pars(?:e|ing) error|failed to (?:parse|load|read) config|"
        r"missing required (?:env|environment variable|config|setting)|"
        r"unknown (?:field|key|property) in config|createcontainerconfigerror",
        re.IGNORECASE,
    ),
    "connection_refused": re.compile(
        r"connection refused|econnrefused|no route to host|could not connect to|"
        r"name or service not known|unknown host",
        re.IGNORECASE,
    ),
    "dependency_timeout": re.compile(
        r"(?:connect|connection|read|upstream|request) timed? ?out|"
        r"(?:context )?deadline exceeded|timeout (?:waiting for|connecting to)",
        re.IGNORECASE,
    ),
    # Envoy / Istio sidecars and gRPC report an unreachable dependency in their own words.
    "upstream_unavailable": re.compile(
        r"upstream connect error|no healthy upstream|disconnect/reset before headers|"
        r"upstream_reset_before_response|remote connection failure|"
        r"code\s*=\s*unavailable|\b50[234]\s+(?:UH|UF|URX|UO|UT)\b",
        re.IGNORECASE,
    ),
    "pool_exhausted": re.compile(
        r"pool exhausted|too many connections|could not obtain (?:a )?connection|"
        r"connection pool (?:is )?(?:full|exhausted)|remaining connection slots are reserved",
        re.IGNORECASE,
    ),
}
CONFIG_CHANGE_PATTERN = re.compile(
    r"\bconfig(?:uration|map)?\b|\bsecret\b|\benv(?:ironment)?\b|values\.ya?ml|\.properties\b",
    re.IGNORECASE,
)
# Kubelet probe failures hit the pod's own port (normal while a new pod starts), so their
# "connection refused" / "timeout" says nothing about a dependency.
_PROBE_FAILURE = re.compile(r"\b(?:readiness|liveness|startup) probe failed\b", re.IGNORECASE)
DEPENDENCY_SIGNATURES = frozenset({"connection_refused", "dependency_timeout", "upstream_unavailable"})
_NEGATION = re.compile(r"\b(?:no|not|without|never)\s+(?:an?\s+)?(?:\w+\s+){0,2}$", re.IGNORECASE)
_TEXT_FIELDS = ("message", "reason", "description", "change", "termination_reason")


def _texts(item: EvidenceItem) -> list[str]:
    texts = [item.summary]
    for field in _TEXT_FIELDS:
        value = item.value.get(field)
        if isinstance(value, str):
            texts.append(value)
    lines = item.value.get("lines")
    if isinstance(lines, list):
        texts.extend(line for line in lines if isinstance(line, str))
    return texts


def _matches(pattern: re.Pattern[str], text: str) -> bool:
    for match in pattern.finditer(text):
        # "no OOMKilled event" must not count as an OOM observation.
        if _NEGATION.search(text[: match.start()]) is None:
            return True
    return False


def line_signatures(text: str) -> frozenset[str]:
    """Return the known signatures found in one line of text."""
    found = frozenset(
        name for name, pattern in SIGNATURE_PATTERNS.items() if _matches(pattern, text)
    )
    return found - DEPENDENCY_SIGNATURES if _PROBE_FAILURE.search(text) else found


def signatures_of(item: EvidenceItem) -> frozenset[str]:
    """Return the known signatures carried by one evidence item."""
    found: set[str] = set()
    declared = item.value.get("signatures")
    if isinstance(declared, list):
        found.update(name for name in declared if name in SIGNATURE_PATTERNS)
    for text in _texts(item):
        found.update(line_signatures(text))
    return frozenset(found)


def is_config_change(item: EvidenceItem) -> bool:
    """True for change records that touch configuration rather than code."""
    if item.evidence_type == "config":
        return True
    if item.evidence_type != "git_change":
        return False
    return any(CONFIG_CHANGE_PATTERN.search(text) for text in _texts(item))


_LIMIT_LOWERED_PHRASE = re.compile(
    r"(?:lower|reduc|decreas|cut|shrink|drop)\w*\s+(?:the\s+)?(?:container\s+|pod\s+)?memory\s+limits?\b|"
    r"\bmemory\s+limits?\s+(?:was\s+|is\s+)?(?:lowered|reduced|decreased|cut)\b",
    re.IGNORECASE,
)
_SIZE = r"(\d+(?:\.\d+)?)\s*(ki|mi|gi|k|m|g)(?:i?b)?\b"
_LIMIT_CHANGE = re.compile(
    r"(?:limits?\.memory|memory\s+limits?)\D{0,20}?" + _SIZE + r"\s*(?:->|=>|→|to)\s*" + _SIZE,
    re.IGNORECASE,
)
_SIZE_UNITS = {"k": 1e3, "m": 1e6, "g": 1e9, "ki": 1024.0, "mi": 1024.0**2, "gi": 1024.0**3}


def memory_limit_lowered(item: EvidenceItem) -> bool:
    """True when a change record says the container memory limit went down."""
    for text in _texts(item):
        if _LIMIT_LOWERED_PHRASE.search(text):
            return True
        for match in _LIMIT_CHANGE.finditer(text):
            before = float(match.group(1)) * _SIZE_UNITS[match.group(2).lower()]
            after = float(match.group(3)) * _SIZE_UNITS[match.group(4).lower()]
            if after < before:
                return True
    return False


def dependency_of(item: EvidenceItem) -> str | None:
    dependency = item.value.get("dependency")
    return dependency if isinstance(dependency, str) and dependency.strip() else None
