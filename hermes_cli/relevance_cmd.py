"""``hermes relevance status|report``."""

from __future__ import annotations

import json
import time
from typing import Any


def _status() -> int:
    from agent.relevance import api_key, load_settings
    from agent.relevance_memory import load_memory_gate_config
    from agent.relevance_skills import load_selection_config
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    settings = load_settings(config)
    skills = load_selection_config(config)
    memory = load_memory_gate_config(config)
    print(f"Backend      TypeSafe {settings.model} · key {'set' if api_key() else 'NOT SET (TYPESAFE_API_KEY)'}")
    on = {True: "on", False: "off"}
    print(f"Skills       selection {on[skills.enabled]} · min {skills.min_score:g} · target "
          f"{skills.target_score:g} · budget {skills.token_budget} · index {skills.index} · rerank "
          f"{on[skills.rerank]} · track_outcomes {on[skills.track_outcomes]}")
    print(f"Memory       write_gate {memory.write_gate} · review_gate {on[memory.review_gate]} · review_events "
          f"{on[memory.review_events]} · recall_filter {on[memory.recall_filter]} · search_rerank "
          f"{on[memory.search_rerank]}")
    print(f"Turn budget  {settings.turn_budget_seconds:g}s per in-turn decision (then it fails open)")
    from agent.relevance_ledger import ledger_path

    print(f"Ledger       {ledger_path()}")
    return 0


def _report(days: float, as_json: bool) -> int:
    from agent.relevance_ledger import ledger_path, read_entries, summarize

    entries = read_entries(since=time.time() - days * 86_400)
    summary = summarize(entries)
    if as_json:
        print(json.dumps({"days": days, "ledger": str(ledger_path()), "features": summary}, indent=2))
        return 0
    if not entries:
        print(f"No relevance decisions in the last {days:g} days ({ledger_path()}).")
        return 0
    print(f"Relevance decisions, last {days:g} days ({len(entries)} entries)\n")
    for feature, s in summary.items():
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items()))
        latency = (f"p50 {s['latency_ms_p50']:.0f} ms, p95 {s['latency_ms_p95']:.0f} ms"
                   if s["latency_ms_p50"] is not None else "no timing")
        print(f"{feature:14} {s['decisions']:5} decisions · {outcomes}")
        print(f"{'':14} {latency} · {s['input_tokens']} input tokens · ~${s['est_cost_usd']:.4f}")
        if feature == "skills" and s.get("most_attached"):
            top = ", ".join(f"{name} ({count})" for name, count in s["most_attached"])
            print(f"{'':14} avg {s['avg_attached']} attached · most attached: {top}")
        if feature == "skill_outcome" and s.get("per_skill"):
            print(f"{'':14} least followed first (skill: attached, mean P(followed)):")
            for name, count, followed in s["per_skill"][:10]:
                print(f"{'':16} {name}: {count}, {followed:.2f}")
    return 0


def cmd_relevance(args: Any) -> int:
    command = getattr(args, "relevance_command", None) or "status"
    if command == "report":
        return _report(args.days, args.json)
    return _status()
