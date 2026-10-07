"""Local ledger of relevance decisions: one JSON line per decision in ``<home>/logs/relevance.jsonl``.

Every gate records what it decided and why (scores, timing, Jev token counts), so the user can see
what the gates did with ``hermes relevance report`` and tune thresholds on real use. Entries hold
skill names, scores and decisions; never message text or memory content (a content hash marks
repeats). ``relevance.ledger: false`` turns it off. The file rotates once at 5 MB.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

LEDGER_NAME = "relevance.jsonl"
MAX_BYTES = 5 * 1024 * 1024
# TypeSafe list price for jev-1.13 input tokens (output tokens are free): $0.042 per million.
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000

_LOCK = threading.Lock()


def ledger_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / LEDGER_NAME


def _enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value

        section = load_config_readonly().get("relevance") or {}
        return is_truthy_value(section.get("ledger") if isinstance(section, dict) else None, default=True)
    except Exception:  # an unreadable config must not silence the ledger
        logger.debug("Could not read relevance.ledger; ledger stays on", exc_info=True)
        return True


def record(feature: str, **fields: Any) -> None:
    """Append one decision. Never raises: a ledger problem must not touch the decision itself."""
    try:
        if not _enabled():
            return
        path = ledger_path()
        line = json.dumps({"ts": round(time.time(), 3), "feature": feature, **fields}, ensure_ascii=False,
                          default=str)
        from hermes_constants import mkdir_under_hermes_home

        with _LOCK:
            mkdir_under_hermes_home(path.parent)  # refuses a deleted profile: the late record is dropped
            if path.exists() and path.stat().st_size > MAX_BYTES:
                path.replace(path.with_suffix(".jsonl.1"))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        logger.debug("Could not append to the relevance ledger", exc_info=True)


def evaluation_fields(evaluation: Any) -> Dict[str, Any]:
    """The cost/latency fields every entry shares, from an ``Evaluation`` (or nothing)."""
    if evaluation is None:
        return {}
    return {"model": evaluation.model, "requests": evaluation.requests, "input_tokens": evaluation.input_tokens,
            "elapsed_ms": evaluation.elapsed_ms}


def read_entries(path: Optional[Path] = None, *, since: Optional[float] = None) -> List[Dict[str, Any]]:
    """Entries from the ledger and its rotated predecessor, oldest first; bad lines are skipped."""
    path = path or ledger_path()
    entries: List[Dict[str, Any]] = []
    for candidate in (path.with_suffix(".jsonl.1"), path):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if isinstance(entry, dict) and (since is None or float(entry.get("ts") or 0) >= since):
                entries.append(entry)
    return entries


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _outcome(entry: Dict[str, Any]) -> str:
    if entry.get("error"):
        return "error"
    feature = entry.get("feature")
    if feature == "skills":
        return "attached" if entry.get("selected") else "none"
    if feature == "write_gate":
        return str(entry.get("decision") or "unknown")
    if feature == "review_gate":
        return "review" if entry.get("run") else "skip"
    if feature == "recall_filter":
        return "trimmed" if (entry.get("kept") or 0) < (entry.get("total") or 0) else "kept_all"
    if feature == "skill_outcome":
        return "followed" if float(entry.get("followed") or 0) >= 0.5 else "ignored"
    return "recorded"


def summarize(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-feature counts, outcomes, latency and cost."""
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        by_feature.setdefault(str(entry.get("feature") or "unknown"), []).append(entry)
    out: Dict[str, Any] = {}
    for feature, rows in sorted(by_feature.items()):
        latency = [float(r["elapsed_ms"]) for r in rows if isinstance(r.get("elapsed_ms"), (int, float))]
        tokens = sum(int(r.get("input_tokens") or 0) for r in rows)
        outcomes: Dict[str, int] = {}
        for r in rows:
            outcomes[_outcome(r)] = outcomes.get(_outcome(r), 0) + 1
        summary: Dict[str, Any] = {
            "decisions": len(rows), "outcomes": outcomes,
            "latency_ms_p50": _pct(latency, 0.5), "latency_ms_p95": _pct(latency, 0.95),
            "input_tokens": tokens, "est_cost_usd": round(tokens * USD_PER_INPUT_TOKEN, 4),
        }
        if feature == "skills":
            counts: Dict[str, int] = {}
            for r in rows:
                for item in r.get("selected") or []:
                    name = item.get("name") if isinstance(item, dict) else str(item)
                    counts[name] = counts.get(name, 0) + 1
            summary["most_attached"] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            sizes = [len(r.get("selected") or []) for r in rows if not r.get("error")]
            summary["avg_attached"] = round(mean(sizes), 2) if sizes else 0
        if feature == "skill_outcome":
            per_skill: Dict[str, List[float]] = {}
            for r in rows:
                per_skill.setdefault(str(r.get("skill")), []).append(float(r.get("followed") or 0))
            summary["per_skill"] = sorted(
                ((name, len(ps), round(mean(ps), 2)) for name, ps in per_skill.items()), key=lambda x: (x[2], -x[1]))
        out[feature] = summary
    return out
