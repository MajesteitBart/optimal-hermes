"""In-process stand-in for TypeSafe's System One API, for relevance-scoring tests (no sockets).

``FakeSystemOne.answer(qid, question, state)`` returns the score (score questions) or the
probability (noul questions) for one question; responses follow the documented shape. The
``fake_systemone`` fixture routes every ``SystemOneClient`` in the process to one fake and sets a
test key; it patches the client's transport seam only, so other HTTP in the test is untouched.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

import httpx
import pytest

Answer = Callable[[str, Dict[str, Any], Any], float]


class FakeSystemOne:
    def __init__(self, answer: Optional[Answer] = None) -> None:
        self.answer: Answer = answer or (lambda qid, question, state: 0.0)
        self.requests: List[Dict[str, Any]] = []
        self.headers: List[Dict[str, str]] = []
        self.urls: List[str] = []
        self.replies: List[Optional[httpx.Response]] = []  # queued overrides; None = answer normally

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        self.urls.append(str(request.url))
        if self.replies:
            reply = self.replies.pop(0)
            if reply is not None:
                return reply
        answers: Dict[str, Any] = {}
        for qid, question in body["questions"].items():
            value = float(self.answer(qid, question, body["state"]))
            if question["type"] == "score":
                top = len(question["criteria"]) - 1
                value = min(max(value, 0.0), float(top))
                answers[qid] = {"type": "score", "score": value, "confidence": 0.9,
                                "probabilities": {str(round(value)): 1.0},
                                "legend": {str(i): c for i, c in enumerate(question["criteria"])}}
            else:
                answers[qid] = {"type": "noul", "noul": min(max(value, 0.0), 1.0)}
        return httpx.Response(200, json={"model": body["model"], "answers": answers,
                                         "usage": {"input_tokens": 100, "output_tokens": 10}})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def questions(self) -> Dict[str, Dict[str, Any]]:
        """Every question sent so far, merged across requests."""
        merged: Dict[str, Dict[str, Any]] = {}
        for body in self.requests:
            merged.update(body["questions"])
        return merged


def skill_scores(scores: Mapping[str, float], *, default: float = 0.0,
                 nouls: Optional[Mapping[str, float]] = None) -> Answer:
    """Score questions by skill name; noul questions by question id (``nouls``), else 0."""

    def answer(qid: str, question: Dict[str, Any], state: Any) -> float:
        if question["type"] == "noul":
            return (nouls or {}).get(qid, 0.0)
        return scores.get(question["instructions"]["skill"]["name"], default)

    return answer


@pytest.fixture
def fake_systemone(monkeypatch) -> FakeSystemOne:
    from agent import relevance_typesafe

    fake = FakeSystemOne()
    original = relevance_typesafe.SystemOneClient.__init__

    def init(self, api_key, **kwargs):
        kwargs["transport"] = kwargs.get("transport") or fake.transport
        kwargs["sleep"] = lambda seconds: None
        original(self, api_key, **kwargs)

    monkeypatch.setattr(relevance_typesafe.SystemOneClient, "__init__", init)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    return fake
