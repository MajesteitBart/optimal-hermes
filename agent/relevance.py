"""Relevance scoring for context management.

Four decisions use it, each opt-in and each owned by a sibling module:

- which skills to attach to a turn (``agent/relevance_skills.py``, ``skills.selection``);
- whether a memory write is worth keeping (``agent/relevance_memory.py``, ``memory.write_gate``);
- whether a background memory review is worth running (same module, ``memory.review_gate``);
- which recalled memories reach the chat (same module, ``memory.recall_filter``).

Scoring runs on TypeSafe's Jev model (``agent/relevance_typesafe.py``): typed questions with
calibrated answers. Code keeps control. Thresholds, budgets and failure handling live in the
features, and the model only answers narrow questions. Every feature sends text to
api.typesafe.ai, so each stays off until configured and needs ``TYPESAFE_API_KEY``.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import httpx

from agent.relevance_typesafe import (
    DEFAULT_MODEL, MAX_REQUEST_TOKENS, MAX_RETRY_WAIT_SECONDS, Evaluation, NoulAnswer, NoulQuestion, Question, RelevanceError,
    RelevanceRequestError, RelevanceResponseError, RelevanceTimeout, RelevanceUnavailable, ScoreAnswer,
    ScoreQuestion, SystemOneClient, build_payload, merge_evaluations, plan_batches,
)

logger = logging.getLogger(__name__)

API_KEY_ENV = "TYPESAFE_API_KEY"

__all__ = [
    "API_KEY_ENV", "Evaluation", "NoulAnswer", "NoulQuestion", "Question", "RelevanceError",
    "RelevanceRequestError", "RelevanceResponseError", "RelevanceSettings", "RelevanceTimeout",
    "RelevanceUnavailable", "agent_config",
    "ScoreAnswer", "ScoreQuestion", "api_key", "build_payload", "describe_error", "error_code",
    "evaluate", "load_settings", "plan_batches",
]


@dataclass(frozen=True)
class RelevanceSettings:
    """The ``relevance:`` config section, shared by every feature."""

    model: str = DEFAULT_MODEL
    timeout_seconds: float = 8.0
    max_retries: int = 2
    max_batch_tokens: int = 12_000
    max_concurrency: int = 4
    turn_budget_seconds: float = 4.0  # wall-clock cap for each decision made inside a user's turn

    def turn_deadline(self) -> float:
        """A ``time.monotonic()`` deadline for one in-turn decision."""
        return time.monotonic() + self.turn_budget_seconds


def _number(raw: Any, default: float, low: float, high: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return min(max(value, low), high)


def load_settings(config: Optional[Mapping[str, Any]] = None) -> RelevanceSettings:
    """Resolve ``relevance:`` from ``config`` (default: the active profile's config.yaml)."""
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    section = config.get("relevance") if isinstance(config, Mapping) else None
    section = section if isinstance(section, Mapping) else {}
    defaults = RelevanceSettings()
    model = section.get("model")
    return RelevanceSettings(
        model=model.strip() if isinstance(model, str) and model.strip() else defaults.model,
        timeout_seconds=_number(section.get("timeout_seconds"), defaults.timeout_seconds, 0.5, 60.0),
        max_retries=int(_number(section.get("max_retries"), defaults.max_retries, 0, 5)),
        max_batch_tokens=int(_number(section.get("max_batch_tokens"), defaults.max_batch_tokens,
                                     2_000, MAX_REQUEST_TOKENS)),
        max_concurrency=int(_number(section.get("max_concurrency"), defaults.max_concurrency, 1, 16)),
        turn_budget_seconds=_number(section.get("turn_budget_seconds"), defaults.turn_budget_seconds, 0.5, 60.0),
    )


def agent_config(agent: Any) -> Mapping[str, Any]:
    """config.yaml of the agent's OWN profile. A gateway build thread can lack the HERMES_HOME
    ContextVar; ambient resolution there would read the launch profile's settings
    (``agent.system_prompt._agent_home`` is the same anchor the skills index uses)."""
    from agent.system_prompt import _agent_home
    from hermes_cli.config import load_config_readonly
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = _agent_home(agent)
    token = set_hermes_home_override(str(home)) if home is not None else None
    try:
        return load_config_readonly()
    finally:
        if token is not None:
            reset_hermes_home_override(token)


def api_key() -> Optional[str]:
    """``TYPESAFE_API_KEY`` for the active profile (scope-aware, never another profile's)."""
    from agent.secret_scope import get_secret

    value = get_secret(API_KEY_ENV)
    return value.strip() if isinstance(value, str) and value.strip() else None


# Relevance state kept on the agent object. All of it belongs to one conversation (config is not cached:
# it is read live, so opting out takes effect at once).
_SESSION_STATE = ("_skill_selection_warned", "_memory_gate_refused", "_memory_lasting_sum")


def sync_session(agent: Any) -> None:
    """Drop per-conversation relevance state once a reused agent moves to another session (/new and
    /resume keep the same AIAgent), so the new session reads the current config and starts clean."""
    try:
        state = vars(agent)
    except TypeError:
        return
    session = getattr(agent, "session_id", None)
    if "_relevance_session" in state and state["_relevance_session"] == session:
        return
    for name in _SESSION_STATE:
        state.pop(name, None)
    state["_relevance_session"] = session


def _call_budget(settings: RelevanceSettings, batches: int) -> float:
    """How long a call without a turn deadline may take: every attempt and retry wait, per round of
    parallel batches. httpx's read timeout restarts with every chunk, so a blocked read cannot be trusted
    to end on its own."""
    per_request = settings.timeout_seconds * (settings.max_retries + 1) + MAX_RETRY_WAIT_SECONDS * settings.max_retries
    rounds = -(-batches // max(1, settings.max_concurrency))
    return per_request * rounds


def evaluate(
    state: Any, questions: Mapping[str, Question], *, settings: Optional[RelevanceSettings] = None,
    key: Optional[str] = None, transport: Optional[httpx.BaseTransport] = None,
    sleep: Optional[Callable[[float], None]] = None, deadline: Optional[float] = None,
) -> Evaluation:
    """Answer every question against ``state``, splitting into parallel requests when needed.

    Raises :class:`RelevanceError` (or a subclass) when any request fails; partial results are
    never returned, because a selection made from half the candidates is quietly wrong."""
    settings = settings or load_settings()
    key = key or api_key()
    if not key:
        raise RelevanceUnavailable(f"{API_KEY_ENV} is not set")
    if not questions:
        return Evaluation(answers={}, model=settings.model, input_tokens=0, output_tokens=0, requests=0, elapsed_ms=0)
    batches = plan_batches(state, questions, max_request_tokens=settings.max_batch_tokens)
    client_kwargs: dict = {"timeout": settings.timeout_seconds, "max_retries": settings.max_retries,
                           "transport": transport, "deadline": deadline}
    if sleep is not None:
        client_kwargs["sleep"] = sleep
    client = SystemOneClient(key, **client_kwargs)
    started = time.monotonic()
    wait_until = deadline if deadline is not None else started + _call_budget(settings, len(batches))
    from agent.memory_provider import ctx_bound

    pool = ThreadPoolExecutor(max_workers=min(len(batches), settings.max_concurrency), thread_name_prefix="relevance")
    try:
        # One context copy per request: a single bound copy cannot be entered by two threads at once.
        futures = [pool.submit(ctx_bound(client.ask), state, batch, model=settings.model) for batch in batches]
        # httpx timeouts are per phase (connect, send, each read), so a request can outlast the budget;
        # the caller stops waiting at the deadline (or the call budget) instead.
        timeout = max(0.0, wait_until - time.monotonic())
        done, pending = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)
        failed = next((future.exception() for future in done if future.exception() is not None), None)
        if failed is not None:
            raise failed
        if pending:
            raise RelevanceTimeout("TypeSafe time budget exhausted")
        parts = [future.result() for future in futures]
    finally:
        # Stragglers are abandoned, not awaited: their phase timeouts and the body deadline end them.
        pool.shutdown(wait=False, cancel_futures=True)
    return merge_evaluations(parts, elapsed_ms=int((time.monotonic() - started) * 1000), model=settings.model)


def error_code(exc: BaseException) -> str:
    """A content-free label for logs that must not carry text: an error body can echo the request."""
    if isinstance(exc, RelevanceUnavailable):
        return "no_key"
    if isinstance(exc, RelevanceTimeout):
        return "timeout"
    if isinstance(exc, RelevanceRequestError):
        return f"http_{exc.status}" if exc.status else "transport"
    if isinstance(exc, RelevanceResponseError):
        return "bad_response"
    return "relevance_error" if isinstance(exc, RelevanceError) else type(exc).__name__


def describe_error(exc: BaseException) -> str:
    """One line for a status message or log; never includes the key."""
    if isinstance(exc, RelevanceUnavailable):
        return f"{API_KEY_ENV} is not set"
    if isinstance(exc, RelevanceError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"
