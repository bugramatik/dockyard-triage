from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dockyard_triage.triage import (
    TriageVerdict,
    build_prompt,
    classify_logs,
    parse_verdict,
    rule_verdict,
    triage_container,
)


OOM_LOGS = [
    "starting worker",
    "loading model weights",
    "Killed process 42 (python) total-vm:2048000kB",
]


class StubCompletions:
    def __init__(self, content: str | None = None, raises: Exception | None = None):
        self.content = content
        self.raises = raises
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def stub_client(content: str | None = None, raises: Exception | None = None):
    completions = StubCompletions(content=content, raises=raises)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


class StubContainer:
    def __init__(self, state: dict, logs: list[str] | None = None):
        self.id = "c0ffee1234"
        self._state = state
        self._logs = logs or []
        self.log_kwargs: dict = {}

    async def show(self, **kwargs):
        return {"Id": self.id, "State": self._state}

    async def log(self, **kwargs):
        self.log_kwargs = kwargs
        return list(self._logs)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Killed process 7 (node) total-vm:900000kB", "out_of_memory"),
        ("KeyError: 'DATABASE_URL'", "missing_config"),
        ("open('/data/x'): permission denied", "permission_denied"),
        ("bind: address already in use", "port_conflict"),
        ("dial tcp 10.0.0.5:5432: connection refused", "dependency_unreachable"),
        ("exec /app/run: exec format error", "image_problem"),
    ],
)
def test_classify_logs_categories(line: str, expected: str) -> None:
    category, evidence = classify_logs(["booting", line])
    assert category == expected
    assert evidence == [line]


def test_classify_logs_unknown() -> None:
    category, evidence = classify_logs(["booting", "ready"])
    assert category == "unknown"
    assert evidence == []


def test_rule_verdict_uses_exit_code_137() -> None:
    verdict = rule_verdict("abc", {"ExitCode": 137, "OOMKilled": True}, ["bye"])
    assert verdict.category == "out_of_memory"
    assert verdict.exit_code == 137
    assert verdict.source == "rules"
    assert "137" in verdict.cause


def test_rule_verdict_strips_ansi() -> None:
    verdict = rule_verdict("abc", {"ExitCode": 1}, ["\x1b[31mpermission denied\x1b[0m"])
    assert verdict.category == "permission_denied"
    assert verdict.evidence == ("permission denied",)


def test_parse_verdict_accepts_json_string() -> None:
    payload = json.dumps(
        {"category": "out_of_memory", "cause": "Memory limit hit.", "evidence": ["Killed"]}
    )
    verdict = parse_verdict("abc", payload, exit_code=137)
    assert isinstance(verdict, TriageVerdict)
    assert verdict.source == "model"
    assert verdict.evidence == ("Killed",)


def test_parse_verdict_rejects_bad_category() -> None:
    with pytest.raises(ValueError):
        parse_verdict("abc", {"category": "cosmic_rays", "cause": "x", "evidence": []})


def test_parse_verdict_rejects_empty_cause() -> None:
    with pytest.raises(ValueError):
        parse_verdict("abc", {"category": "unknown", "cause": "  ", "evidence": []})


def test_build_prompt_mentions_state_and_logs() -> None:
    prompt = build_prompt("abc", {"ExitCode": 1, "Status": "exited"}, ["boom"])
    assert "\"exit_code\": 1" in prompt
    assert "boom" in prompt


@pytest.mark.asyncio
async def test_triage_container_uses_model() -> None:
    client, completions = stub_client(
        json.dumps(
            {
                "category": "out_of_memory",
                "cause": "The worker was killed after passing its memory limit.",
                "evidence": [OOM_LOGS[-1]],
            }
        )
    )
    container = StubContainer({"ExitCode": 137, "OOMKilled": True}, OOM_LOGS)
    verdict = await triage_container(container, client=client)
    assert verdict.source == "model"
    assert verdict.category == "out_of_memory"
    assert completions.calls[0]["response_format"]["type"] == "json_schema"
    assert container.log_kwargs["stdout"] is True
    assert container.log_kwargs["stderr"] is True


@pytest.mark.asyncio
async def test_triage_container_falls_back_on_bad_json() -> None:
    client, _ = stub_client("not json at all")
    container = StubContainer({"ExitCode": 1}, ["KeyError: 'API_TOKEN'"])
    verdict = await triage_container(container, client=client)
    assert verdict.source == "rules"
    assert verdict.category == "missing_config"


@pytest.mark.asyncio
async def test_triage_container_falls_back_when_model_raises() -> None:
    client, _ = stub_client(raises=RuntimeError("rate limited"))
    container = StubContainer({"ExitCode": 1}, ["bind: address already in use"])
    verdict = await triage_container(container, client=client)
    assert verdict.source == "rules"
    assert verdict.category == "port_conflict"


@pytest.mark.asyncio
async def test_triage_container_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    container = StubContainer({"ExitCode": 2}, ["exec format error"])
    verdict = await triage_container(container)
    assert verdict.source == "rules"
    assert verdict.category == "image_problem"
    assert verdict.as_dict()["exit_code"] == 2
