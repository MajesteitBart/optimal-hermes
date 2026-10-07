"""TypeSafe System One client: request shape, batching, retries, strict response validation."""

from __future__ import annotations

import httpx
import pytest

from agent.relevance import RelevanceSettings, evaluate
from agent.relevance_typesafe import (
    TYPESAFE_ENDPOINT, NoulAnswer, NoulQuestion, RelevanceError, RelevanceRequestError, RelevanceResponseError,
    RelevanceUnavailable, ScoreAnswer, ScoreQuestion, SystemOneClient, parse_answers, plan_batches,
)
from tests._fixtures.typesafe_fake import FakeSystemOne

LEVELS = ("low", "mid", "high")


def _score(qid="q", value=1.0):
    return {"type": "score", "score": value, "confidence": 0.5, "probabilities": {"1": 1.0}}


def _client(fake, **kw):
    sleeps = []
    return SystemOneClient("k", transport=fake.transport, sleep=sleeps.append, **kw), sleeps


class TestParseAnswers:
    QUESTIONS = {"a": ScoreQuestion("rate", LEVELS), "b": NoulQuestion("yes?")}

    def test_valid_answers_parse_to_typed_values(self):
        payload = {"model": "jev-1.13.0", "answers": {"a": _score(value=1.4), "b": {"type": "noul", "noul": 0.8}},
                   "usage": {"input_tokens": 12, "output_tokens": 3}}
        answers, model, inp, out = parse_answers(payload, self.QUESTIONS)
        assert isinstance(answers["a"], ScoreAnswer) and answers["a"].score == pytest.approx(1.4)
        assert isinstance(answers["b"], NoulAnswer) and answers["b"].value == pytest.approx(0.8)
        assert (model, inp, out) == ("jev-1.13.0", 12, 3)

    @pytest.mark.parametrize("answers", [
        {"a": _score()},  # missing b
        {"a": _score(), "b": {"type": "noul", "noul": 0.1}, "c": {"type": "noul", "noul": 0.1}},  # extra id
        {"a": {"type": "noul", "noul": 0.5}, "b": {"type": "noul", "noul": 0.1}},  # wrong type
        {"a": _score(value=2.5), "b": {"type": "noul", "noul": 0.1}},  # above top level
        {"a": _score(value=float("nan")), "b": {"type": "noul", "noul": 0.1}},
        {"a": _score(value=True), "b": {"type": "noul", "noul": 0.1}},  # bool is not a number
        {"a": _score(), "b": {"type": "noul", "noul": 1.2}},
        {"a": {**_score(), "probabilities": {"7": 1.0}}, "b": {"type": "noul", "noul": 0.1}},
    ])
    def test_malformed_answers_fail_closed(self, answers):
        with pytest.raises(RelevanceResponseError):
            parse_answers({"answers": answers}, self.QUESTIONS)

    def test_non_object_body_fails(self):
        with pytest.raises(RelevanceResponseError):
            parse_answers(["nope"], self.QUESTIONS)


class TestPlanBatches:
    def test_splits_in_order_under_the_request_budget(self):
        questions = {f"q{i}": NoulQuestion("x" * 400) for i in range(10)}
        batches = plan_batches("state", questions, max_request_tokens=400)
        assert len(batches) > 1
        assert [qid for batch in batches for qid in batch] == list(questions)

    def test_refuses_a_state_too_large_for_any_question(self):
        with pytest.raises(RelevanceError):
            plan_batches("s" * 200_000, {"q": NoulQuestion("x")})


class TestClient:
    def test_posts_bearer_payload_to_the_fixed_endpoint(self):
        fake = FakeSystemOne(lambda qid, q, state: 0.7)
        client, _ = _client(fake)
        result = client.ask({"text": "hi"}, {"b": NoulQuestion("yes?")}, model="jev-1.13.0")
        assert fake.urls == [TYPESAFE_ENDPOINT]
        assert fake.headers[0]["authorization"] == "Bearer k"
        assert fake.requests[0] == {"model": "jev-1.13.0", "state": {"text": "hi"},
                                    "questions": {"b": {"type": "noul", "instructions": "yes?"}}}
        assert result.answers["b"].value == pytest.approx(0.7)

    def test_retries_rate_limits_honoring_retry_after(self):
        fake = FakeSystemOne(lambda qid, q, state: 0.5)
        fake.replies = [httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(529), None]
        client, sleeps = _client(fake, max_retries=2)
        assert client.ask("s", {"b": NoulQuestion("?")}).answers["b"].value == pytest.approx(0.5)
        assert sleeps[0] == pytest.approx(1.0) and len(sleeps) == 2

    def test_long_retry_after_is_not_waited_out(self):
        fake = FakeSystemOne()
        fake.replies = [httpx.Response(429, headers={"Retry-After": "120"})]
        client, sleeps = _client(fake, max_retries=3)
        with pytest.raises(RelevanceRequestError) as exc:
            client.ask("s", {"b": NoulQuestion("?")})
        assert exc.value.status == 429 and sleeps == [] and len(fake.requests) == 1

    def test_bad_key_fails_without_retry(self):
        fake = FakeSystemOne()
        fake.replies = [httpx.Response(401, text="no")]
        client, _ = _client(fake, max_retries=3)
        with pytest.raises(RelevanceRequestError, match="TYPESAFE_API_KEY"):
            client.ask("s", {"b": NoulQuestion("?")})
        assert len(fake.requests) == 1

    def test_redirects_are_never_followed(self):
        fake = FakeSystemOne()
        fake.replies = [httpx.Response(307, headers={"Location": "https://elsewhere.example/steal"})]
        client, _ = _client(fake)
        with pytest.raises(RelevanceRequestError, match="redirect"):
            client.ask("s", {"b": NoulQuestion("?")})
        assert fake.urls == [TYPESAFE_ENDPOINT]

    def test_transport_errors_retry_then_raise(self):
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ConnectError("down")

        client = SystemOneClient("k", transport=httpx.MockTransport(handler), max_retries=1, sleep=lambda s: None)
        with pytest.raises(RelevanceRequestError, match="ConnectError"):
            client.ask("s", {"b": NoulQuestion("?")})
        assert len(calls) == 2

    def test_non_json_body_is_a_response_error(self):
        fake = FakeSystemOne()
        fake.replies = [httpx.Response(200, text="<html>")]
        client, _ = _client(fake)
        with pytest.raises(RelevanceResponseError):
            client.ask("s", {"b": NoulQuestion("?")})

    def test_missing_key_is_unavailable(self):
        with pytest.raises(RelevanceUnavailable):
            SystemOneClient("")


class TestEvaluate:
    def test_parallel_batches_merge_answers_and_usage(self, fake_systemone):
        fake_systemone.answer = lambda qid, q, state: int(qid[1:]) % 2
        questions = {f"q{i}": NoulQuestion("x" * 3000) for i in range(12)}
        result = evaluate("state", questions, settings=RelevanceSettings(max_batch_tokens=2000))
        assert result.requests == len(fake_systemone.requests) > 1
        assert set(result.answers) == set(questions)
        assert result.input_tokens == 100 * result.requests
        assert [result.answers[f"q{i}"].value for i in range(4)] == [0.0, 1.0, 0.0, 1.0]

    def test_one_failed_batch_fails_the_whole_evaluation(self, fake_systemone):
        fake_systemone.replies = [None, httpx.Response(422, text="bad")]
        questions = {f"q{i}": NoulQuestion("x" * 3000) for i in range(12)}
        with pytest.raises(RelevanceRequestError):
            evaluate("state", questions, settings=RelevanceSettings(max_batch_tokens=2000, max_retries=0))

    def test_no_key_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(RelevanceUnavailable):
            evaluate("s", {"b": NoulQuestion("?")}, settings=RelevanceSettings())

    def test_no_questions_sends_nothing(self, fake_systemone):
        assert evaluate("s", {}, settings=RelevanceSettings()).requests == 0
        assert fake_systemone.requests == []
