"""``hermes memory audit``: judge the built-in memory you already have, entry by entry.

Runs the write-gate questions (``agent.relevance_memory``) over every MEMORY.md / USER.md entry,
each against the rest of its store: duplicates, entries one another supersedes, task progress,
procedures that belong in a skill, low lasting value, and entries filed in the wrong store. The
report is read-only; nothing is changed. It also lists the entries with the least lasting value,
which is the answer to "memory is full: what matters least?".
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List


def audit_store(store: Any, targets: List[str], *, transport: Any = None) -> List[Dict[str, Any]]:
    """One row per entry: ``target``, ``entry``, ``keep``, ``reason``, ``lasting_value``, ``store_hint``."""
    from agent.relevance_memory import all_entries, assess_writes

    everything = all_entries(store)
    rows: List[Dict[str, Any]] = []
    for target in targets:
        entries = [str(e) for e in (store._entries_for(target) or [])]
        # Each entry against all others: drop only the audited occurrence, so identical copies count.
        offset = 0 if target == "memory" else len(store._entries_for("memory") or [])
        candidates = [(entry, target, everything[:offset + i] + everything[offset + i + 1:])
                      for i, entry in enumerate(entries)]
        verdicts = assess_writes(candidates, user_message="", transport=transport) if candidates else []
        for entry, verdict in zip(entries, verdicts):
            rows.append({
                "target": target, "entry": entry, "keep": verdict.keep, "reason": verdict.reason,
                "lasting_value": round((verdict.scores.get("durable", 0) + verdict.scores.get("personal", 0)) / 2, 3),
                "store_hint": verdict.store_hint, "supersedes": verdict.supersedes,
                "scores": {k: round(v, 3) for k, v in verdict.scores.items()},
            })
    return rows


def _print_report(rows: List[Dict[str, Any]]) -> None:
    flagged = [r for r in rows if not r["keep"] or r["store_hint"]]
    print(f"{len(rows)} entries audited · {sum(not r['keep'] for r in rows)} to review · "
          f"{sum(1 for r in rows if r['store_hint'])} possibly in the wrong store\n")
    for r in flagged:
        label = "USER.md" if r["target"] == "user" else "MEMORY.md"
        print(f"[{label}] {r['entry'][:140]}")
        if not r["keep"]:
            print(f"    review: {r['reason']}")
        if r["store_hint"]:
            print(f"    belongs in {'USER.md' if r['store_hint'] == 'user' else 'MEMORY.md'}?")
    least = sorted(rows, key=lambda r: r["lasting_value"])[:5]
    if least:
        print("\nLeast lasting value (first to drop when memory is full):")
        for r in least:
            print(f"  {r['lasting_value']:.2f}  {r['entry'][:120]}")
    if not flagged:
        print("Nothing to review.")


def cmd_memory_audit(args: Any) -> int:
    from agent.relevance import RelevanceError, api_key, describe_error
    from tools.memory_tool import load_on_disk_store

    if not api_key():
        print("error: TYPESAFE_API_KEY is not set; the audit scores entries with Jev", file=sys.stderr)
        return 1
    store = load_on_disk_store()
    target = getattr(args, "target", "all") or "all"
    targets = [t for t in ("memory", "user") if target in ("all", t) and store.target_enabled(t)]
    try:
        rows = audit_store(store, targets)
    except RelevanceError as exc:
        print(f"error: audit failed: {describe_error(exc)}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps({"entries": rows}, indent=2, ensure_ascii=False))
    else:
        _print_report(rows)
    return 0
