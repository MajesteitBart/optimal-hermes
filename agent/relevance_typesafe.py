"""TypeSafe System One client: the Jev model behind relevance scoring.

One endpoint, ``POST https://api.typesafe.ai/v1/systemone``: a ``state`` plus a map of typed
questions returns one typed answer per question, with calibrated probabilities. Hermes asks two
kinds: ``score`` (an ordered rubric, answered with a probability-weighted level) and ``noul``
(a yes/no question, answered with the probability of yes).

The origin is fixed and redirects are refused, so the bearer key never reaches another host.
Responses are validated strictly: a missing, extra, mistyped or out-of-range answer fails the
whole request rather than feeding a guess into a selection decision. Callers decide what a
failure means for their feature; this module never substitutes a heuristic.

API: https://docs.typesafe.ai/api.md · limits: https://docs.typesafe.ai/models.md
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import httpx

from agent.model_metadata import estimate_tokens_rough
from agent.retry_utils import parse_retry_after_seconds

logger = logging.getLogger(__name__)

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"

# Documented limits: 64k tokens per request, 32k for the state plus the longest question.
# estimate_tokens_rough is not TypeSafe's tokenizer, so batches stay well under both.
MAX_REQUEST_TOKENS = 40_000
MAX_STATE_PLUS_QUESTION_TOKENS = 24_000

# 429 = rate limited, 529 = overloaded (both documented as retryable); 502-504 are proxy blips.
RETRY_STATUSES = frozenset({429, 502, 503, 504, 529})
# Scoring runs inside a user turn: a long Retry-After means "not now", not "wait a minute".
MAX_RETRY_WAIT_SECONDS = 4.0
# An attempt needs at least this much of the time budget left to be worth starting.
MIN_ATTEMPT_SECONDS = 0.3
MAX_RESPONSE_BYTES = 8_000_000  # a full batch answers in well under 1 MB


class RelevanceError(RuntimeError):
    """Scoring could not produce a trustworthy answer."""


class RelevanceUnavailable(RelevanceError):
    """No credential, or the feature is not configured."""


class RelevanceRequestError(RelevanceError):
    """The HTTP exchange failed (transport error, non-retryable status, retries exhausted)."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class RelevanceTimeout(RelevanceRequestError):
    """The time budget ran out before an answer arrived (per-turn callers set one)."""


class RelevanceResponseError(RelevanceError):
    """The service answered, but not with what was asked."""


@dataclass(frozen=True)
class ScoreQuestion:
    """Rate the state on an ordered rubric; ``criteria[i]`` describes level ``i``."""

    instructions: Any
    criteria: Tuple[Any, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.criteria) <= 10:
            raise ValueError("a score question needs 2-10 levels")

    def wire(self) -> Dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass(frozen=True)
class NoulQuestion:
    """A yes/no question; the answer is the probability of yes."""

    instructions: Any
    criteria: Optional[Mapping[str, Any]] = None

    def wire(self) -> Dict[str, Any]:
        wire: Dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            wire["criteria"] = dict(self.criteria)
        return wire


Question = Union[ScoreQuestion, NoulQuestion]


@dataclass(frozen=True)
class ScoreAnswer:
    score: float  # expected level, 0 .. len(criteria) - 1; may land between levels
    confidence: Optional[float]
    probabilities: Dict[int, float]


@dataclass(frozen=True)
class NoulAnswer:
    value: float  # probability of yes


Answer = Union[ScoreAnswer, NoulAnswer]


@dataclass(frozen=True)
class Evaluation:
    answers: Dict[str, Answer]
    model: str
    input_tokens: int
    output_tokens: int
    requests: int
    elapsed_ms: int


def _finite(value: Any) -> Optional[float]:
    """A real number as float; ``None`` for bools, NaN, infinities and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _in_range(value: Any, low: float, high: float, what: str) -> float:
    number = _finite(value)
    if number is None or not (low - 1e-6 <= number <= high + 1e-6):
        raise RelevanceResponseError(f"{what} is not a number in [{low:g}, {high:g}]: {value!r}")
    return min(max(number, low), high)


def _parse_score(qid: str, raw: Mapping[str, Any], question: ScoreQuestion) -> ScoreAnswer:
    top = len(question.criteria) - 1
    score = _in_range(raw.get("score"), 0.0, float(top), f"score for {qid!r}")
    confidence = raw.get("confidence")
    confidence = None if confidence is None else _in_range(confidence, 0.0, 1.0, f"confidence for {qid!r}")
    probabilities: Dict[int, float] = {}
    raw_probs = raw.get("probabilities")
    if isinstance(raw_probs, Mapping):
        for level, prob in raw_probs.items():
            try:
                index = int(level)
            except (TypeError, ValueError):
                raise RelevanceResponseError(f"probability level for {qid!r} is not an index: {level!r}") from None
            if not 0 <= index <= top:
                raise RelevanceResponseError(f"probability level {index} for {qid!r} is outside 0..{top}")
            probabilities[index] = _in_range(prob, 0.0, 1.0, f"probability for {qid!r}")
    return ScoreAnswer(score=score, confidence=confidence, probabilities=probabilities)


def parse_answers(payload: Any, questions: Mapping[str, Question]) -> Tuple[Dict[str, Answer], str, int, int]:
    """Validate one response against the questions it answers.

    Returns ``(answers, model, input_tokens, output_tokens)``; raises :class:`RelevanceResponseError`
    on any mismatch. The answer ids must equal the question ids exactly."""
    if not isinstance(payload, Mapping):
        raise RelevanceResponseError("response body is not a JSON object")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise RelevanceResponseError("response has no 'answers' object")
    missing = sorted(set(questions) - set(raw_answers))
    extra = sorted(set(raw_answers) - set(questions))
    if missing or extra:
        raise RelevanceResponseError(f"answer ids do not match the questions (missing={missing[:5]}, extra={extra[:5]})")
    answers: Dict[str, Answer] = {}
    for qid, question in questions.items():
        raw = raw_answers[qid]
        if not isinstance(raw, Mapping):
            raise RelevanceResponseError(f"answer {qid!r} is not an object")
        expected = "score" if isinstance(question, ScoreQuestion) else "noul"
        if raw.get("type") != expected:
            raise RelevanceResponseError(f"answer {qid!r} has type {raw.get('type')!r}, expected {expected!r}")
        if isinstance(question, ScoreQuestion):
            answers[qid] = _parse_score(qid, raw, question)
        else:
            answers[qid] = NoulAnswer(value=_in_range(raw.get("noul"), 0.0, 1.0, f"noul for {qid!r}"))
    model = payload.get("model")
    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
    tokens = [_finite(usage.get(key)) for key in ("input_tokens", "output_tokens")]
    return answers, str(model) if isinstance(model, str) else "", int(tokens[0] or 0), int(tokens[1] or 0)


def _wire_tokens(value: Any) -> int:
    return estimate_tokens_rough(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def plan_batches(
    state: Any, questions: Mapping[str, Question], *, max_request_tokens: int = MAX_REQUEST_TOKENS,
    max_state_plus_question: int = MAX_STATE_PLUS_QUESTION_TOKENS,
) -> List[Dict[str, Question]]:
    """Split ``questions`` into request-sized groups (insertion order kept).

    Every request resends the state, so a group holds as many questions as fit beside it. Raises
    :class:`RelevanceError` when the state plus a single question already exceeds the limit: the
    caller must send less state, never a silently cut question."""
    state_tokens = _wire_tokens(state)
    batches: List[Dict[str, Question]] = []
    current: Dict[str, Question] = {}
    current_tokens = state_tokens
    for qid, question in questions.items():
        q_tokens = _wire_tokens({qid: question.wire()})
        if state_tokens + q_tokens > max_state_plus_question:
            raise RelevanceError(
                f"question {qid!r} plus the state is ~{state_tokens + q_tokens} tokens, over the "
                f"{max_state_plus_question}-token limit; send less context")
        if current and current_tokens + q_tokens > max_request_tokens:
            batches.append(current)
            current, current_tokens = {}, state_tokens
        current[qid] = question
        current_tokens += q_tokens
    if current:
        batches.append(current)
    return batches


def build_payload(state: Any, questions: Mapping[str, Question], model: str) -> Dict[str, Any]:
    return {"model": model, "state": state, "questions": {qid: q.wire() for qid, q in questions.items()}}


def _status_message(response: httpx.Response) -> str:
    """The error for a non-200 reply. Never quotes the body: it can echo the request (memory, messages),
    and this text reaches logs, the ledger and chat warnings."""
    status = response.status_code
    if status == 401:
        return "TypeSafe rejected the API key (HTTP 401); check TYPESAFE_API_KEY"
    if 300 <= status < 400:
        return f"TypeSafe answered with a redirect (HTTP {status}); refusing to follow it with credentials"
    return f"TypeSafe request failed (HTTP {status})"


class SystemOneClient:
    """Synchronous client for one TypeSafe account. ``transport`` is the test seam."""

    def __init__(
        self, api_key: str, *, timeout: float = 8.0, max_retries: int = 2,
        transport: Optional[httpx.BaseTransport] = None, sleep: Callable[[float], None] = time.sleep,
        deadline: Optional[float] = None,
    ) -> None:
        """``deadline`` (a ``time.monotonic()`` value) bounds every attempt and retry wait; past it the
        request fails with :class:`RelevanceTimeout`."""
        if not api_key:
            raise RelevanceUnavailable("TYPESAFE_API_KEY is not set")
        self._api_key = api_key
        self._timeout = max(0.5, float(timeout))
        self._max_retries = max(0, int(max_retries))
        self._transport = transport
        self._sleep = sleep
        self._deadline = deadline

    def _attempt_timeout(self) -> float:
        if self._deadline is None:
            return self._timeout
        remaining = self._deadline - time.monotonic()
        if remaining < MIN_ATTEMPT_SECONDS:
            raise RelevanceTimeout("TypeSafe time budget exhausted")
        return min(self._timeout, remaining)

    def _retry_after(self, delay: Optional[float]) -> bool:
        """Wait ``delay`` before another attempt; False when there is no retry to make (no delay, or the
        time budget cannot cover the wait plus an attempt)."""
        if delay is None:
            return False
        if self._deadline is not None and delay + MIN_ATTEMPT_SECONDS > self._deadline - time.monotonic():
            return False
        self._sleep(delay)
        return True

    def ask(self, state: Any, questions: Mapping[str, Question], *, model: str = DEFAULT_MODEL) -> Evaluation:
        """One request: every question answered against ``state``."""
        started = time.monotonic()
        body = self._post(build_payload(state, questions, model))
        answers, served_model, input_tokens, output_tokens = parse_answers(body, questions)
        return Evaluation(answers=answers, model=served_model or model, input_tokens=input_tokens,
                          output_tokens=output_tokens, requests=1,
                          elapsed_ms=int((time.monotonic() - started) * 1000))

    def _retry_delay(self, attempt: int, response: Optional[httpx.Response]) -> Optional[float]:
        """Seconds to wait before attempt ``attempt + 1``; ``None`` when the server asks for longer
        than an interactive turn should wait."""
        if response is not None:
            hinted = parse_retry_after_seconds(response.headers)
            if hinted is not None:
                return hinted if hinted <= MAX_RETRY_WAIT_SECONDS else None
        base = min(MAX_RETRY_WAIT_SECONDS, 0.25 * (2 ** (attempt - 1)))
        return base + random.uniform(0, base / 2)

    def _post(self, payload: Dict[str, Any]) -> Any:
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json",
                   "User-Agent": "hermes-agent"}
        attempts = self._max_retries + 1
        with httpx.Client(follow_redirects=False, transport=self._transport) as client:
            for attempt in range(1, attempts + 1):
                retry_delay = self._retry_delay(attempt, None) if attempt < attempts else None
                try:
                    phase_timeout = self._attempt_timeout()
                    # httpx's timeout restarts with every chunk; timeout_seconds bounds the attempt as a whole.
                    until = time.monotonic() + phase_timeout
                    with client.stream("POST", TYPESAFE_ENDPOINT, json=payload, headers=headers,
                                       timeout=httpx.Timeout(phase_timeout)) as response:
                        content = self._read_body(response, until)
                except httpx.TransportError as exc:
                    logger.debug("TypeSafe transport error (attempt %d/%d): %s", attempt, attempts, exc)
                    if self._retry_after(retry_delay):
                        continue
                    error = RelevanceTimeout if isinstance(exc, httpx.TimeoutException) else RelevanceRequestError
                    raise error(f"TypeSafe request failed: {type(exc).__name__}") from exc
                if response.status_code == 200:
                    try:
                        return json.loads(content)
                    except ValueError as exc:
                        raise RelevanceResponseError("TypeSafe returned a body that is not JSON") from exc
                # No body: it can echo the request (memory, messages) into a log someone else reads.
                logger.debug("TypeSafe HTTP %d (attempt %d/%d)", response.status_code, attempt, attempts)
                retryable = response.status_code in RETRY_STATUSES and attempt < attempts
                if retryable and self._retry_after(self._retry_delay(attempt, response)):
                    continue
                raise RelevanceRequestError(_status_message(response), status=response.status_code)
        raise AssertionError("unreachable: the last attempt always returns or raises")

    def _read_body(self, response: httpx.Response, until: float) -> bytes:
        """The whole body, abandoned at ``until`` (the attempt's timeout, capped by the turn deadline) or
        past MAX_RESPONSE_BYTES: a server that trickles or streams endlessly cannot hold the call open."""
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise RelevanceResponseError(f"TypeSafe response exceeds {MAX_RESPONSE_BYTES} bytes")
            if time.monotonic() > until:
                raise RelevanceTimeout("TypeSafe request took longer than its timeout")
        return b"".join(chunks)


def merge_evaluations(parts: Sequence[Evaluation], *, elapsed_ms: int, model: str) -> Evaluation:
    answers: Dict[str, Answer] = {}
    for part in parts:
        answers.update(part.answers)
    served = next((p.model for p in parts if p.model), model)
    return Evaluation(answers=answers, model=served,
                      input_tokens=sum(p.input_tokens for p in parts),
                      output_tokens=sum(p.output_tokens for p in parts),
                      requests=sum(p.requests for p in parts), elapsed_ms=elapsed_ms)
