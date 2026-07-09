from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence


if TYPE_CHECKING:
    from .containers import DockerContainer


CATEGORIES = (
    "missing_config",
    "image_problem",
    "out_of_memory",
    "permission_denied",
    "port_conflict",
    "dependency_unreachable",
    "application_error",
    "unknown",
)

DEFAULT_MODEL = "gpt-4o-mini"
MAX_LOG_LINES = 200

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "out_of_memory",
        re.compile(
            r"out of memory|oomkill|oom-kill|killed process|memory cgroup out of memory",
            re.IGNORECASE,
        ),
    ),
    (
        "missing_config",
        re.compile(
            r"environment variable .* (is )?(not set|missing|required)"
            r"|missing (required )?(env|environment)"
            r"|keyerror: '?[A-Z_]{3,}'?"
            r"|no such file or directory: '/run/secrets",
            re.IGNORECASE,
        ),
    ),
    (
        "permission_denied",
        re.compile(r"permission denied|operation not permitted|eacces", re.IGNORECASE),
    ),
    (
        "port_conflict",
        re.compile(
            r"address already in use|port is already allocated|bind: address already",
            re.IGNORECASE,
        ),
    ),
    (
        "dependency_unreachable",
        re.compile(
            r"connection refused|no such host|temporary failure in name resolution"
            r"|could not resolve host|connection timed out",
            re.IGNORECASE,
        ),
    ),
    (
        "image_problem",
        re.compile(
            r"exec format error|no such file or directory: '?/bin"
            r"|manifest unknown|pull access denied|executable file not found",
            re.IGNORECASE,
        ),
    ),
)

_CAUSE_BY_CATEGORY = {
    "out_of_memory": "The kernel killed the process after it exceeded its memory limit.",
    "missing_config": "A required environment variable or secret was not provided.",
    "permission_denied": "The process tried to touch a file or socket it may not access.",
    "port_conflict": "The container tried to bind a port that is already taken.",
    "dependency_unreachable": "A service the container depends on did not answer.",
    "image_problem": "The image or its entrypoint does not match the host it runs on.",
    "application_error": "The application raised an error and exited on its own.",
    "unknown": "The logs do not point at a single cause.",
}


@dataclass(frozen=True)
class TriageVerdict:
    """A short, structured explanation of why a container stopped."""

    container_id: str
    category: str
    cause: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    source: str = "rules"
    exit_code: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "container_id": self.container_id,
            "category": self.category,
            "cause": self.cause,
            "evidence": list(self.evidence),
            "source": self.source,
            "exit_code": self.exit_code,
        }

    def __str__(self) -> str:
        return f"[{self.category}] {self.cause}"


class TriageError(RuntimeError):
    """Raised when a container's state cannot be read for triage."""


def _normalise(lines: Sequence[str]) -> list[str]:
    """Strip ANSI colour codes and trailing whitespace from log lines."""
    cleaned: list[str] = []
    for raw in lines:
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw).rstrip()
        if text:
            cleaned.append(text)
    return cleaned


def _tail(lines: Sequence[str], limit: int = MAX_LOG_LINES) -> list[str]:
    if limit <= 0:
        return []
    return list(lines[-limit:])


def classify_logs(lines: Sequence[str]) -> tuple[str, list[str]]:
    """Classify log lines with ordered rules.

    Returns the category and the lines that matched, in log order.
    """
    evidence: list[str] = []
    matched: str | None = None
    for line in lines:
        for category, pattern in _RULES:
            if pattern.search(line):
                if matched is None:
                    matched = category
                if line not in evidence:
                    evidence.append(line)
                break
    if matched is None:
        if any(re.search(r"traceback|exception|error", line, re.IGNORECASE) for line in lines):
            return "application_error", [
                line
                for line in lines
                if re.search(r"traceback|exception|error", line, re.IGNORECASE)
            ][-3:]
        return "unknown", []
    return matched, evidence[-5:]


def _exit_code(state: Mapping[str, Any]) -> int | None:
    code = state.get("ExitCode")
    if isinstance(code, int):
        return code
    return None


def rule_verdict(
    container_id: str,
    state: Mapping[str, Any],
    lines: Sequence[str],
) -> TriageVerdict:
    """Build a verdict from the container state and logs without a model."""
    cleaned = _normalise(lines)
    category, evidence = classify_logs(cleaned)
    exit_code = _exit_code(state)
    if category == "unknown" and exit_code == 137:
        category = "out_of_memory"
    cause = _CAUSE_BY_CATEGORY[category]
    if exit_code is not None and category != "unknown":
        cause = f"{cause} Exit code {exit_code}."
    return TriageVerdict(
        container_id=container_id,
        category=category,
        cause=cause,
        evidence=tuple(evidence),
        source="rules",
        exit_code=exit_code,
    )


_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["category", "cause", "evidence"],
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "cause": {"type": "string", "maxLength": 200},
        "evidence": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string"},
        },
    },
}

_SYSTEM_PROMPT = (
    "You triage stopped Docker containers. Read the exit state and the last log "
    "lines and answer with one category, one sentence of at most 30 words that "
    "names the cause, and up to five log lines copied verbatim that support it. "
    "Never invent a log line."
)


def build_prompt(
    container_id: str,
    state: Mapping[str, Any],
    lines: Sequence[str],
) -> str:
    """Render the user message sent to the model."""
    summary = {
        "container_id": container_id,
        "status": state.get("Status"),
        "exit_code": state.get("ExitCode"),
        "oom_killed": state.get("OOMKilled"),
        "error": state.get("Error") or "",
    }
    body = "\n".join(lines) if lines else "(no log output)"
    return (
        f"Container state: {json.dumps(summary, sort_keys=True)}\n"
        f"Last {len(lines)} log lines:\n{body}"
    )


def parse_verdict(
    container_id: str,
    payload: str | Mapping[str, Any],
    exit_code: int | None = None,
) -> TriageVerdict:
    """Validate a model reply into a TriageVerdict.

    Raises ValueError when the reply is not the expected JSON object.
    """
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    category = data.get("category")
    cause = data.get("cause")
    evidence = data.get("evidence", [])
    if category not in CATEGORIES:
        raise ValueError(f"unknown category: {category!r}")
    if not isinstance(cause, str) or not cause.strip():
        raise ValueError("cause must be a non-empty string")
    if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
        raise ValueError("evidence must be a list of strings")
    return TriageVerdict(
        container_id=container_id,
        category=category,
        cause=cause.strip(),
        evidence=tuple(evidence[:5]),
        source="model",
        exit_code=exit_code,
    )


async def _ask_model(
    client: Any,
    model: str,
    container_id: str,
    state: Mapping[str, Any],
    lines: Sequence[str],
) -> TriageVerdict:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "triage_verdict", "schema": _VERDICT_SCHEMA},
        },
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(container_id, state, lines)},
        ],
    )
    content = response.choices[0].message.content or ""
    return parse_verdict(container_id, content, _exit_code(state))


def _default_client() -> Any | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None
    return AsyncOpenAI()


async def triage_container(
    container: DockerContainer,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_lines: int = MAX_LOG_LINES,
) -> TriageVerdict:
    """Explain why a container stopped.

    Reads the container's inspect state and recent stdout and stderr, asks the
    model for a structured verdict when a client is available, and falls back
    to the rule-based classifier when there is no client or the reply does not
    parse.
    """
    try:
        info = await container.show()
    except Exception as exc:  # noqa: BLE001 - surfaced as TriageError
        raise TriageError(f"cannot inspect container {container.id}: {exc}") from exc

    state = info.get("State") or {}
    try:
        lines = await container.log(stdout=True, stderr=True, tail=max_lines)
    except Exception:  # noqa: BLE001 - a container without logs is still triageable
        lines = []

    cleaned = _tail(_normalise(lines), max_lines)
    fallback = rule_verdict(container.id, state, cleaned)

    if client is None:
        client = _default_client()
    if client is None:
        return fallback

    try:
        return await _ask_model(client, model, container.id, state, cleaned)
    except Exception:  # noqa: BLE001 - any model failure falls back to rules
        return fallback
