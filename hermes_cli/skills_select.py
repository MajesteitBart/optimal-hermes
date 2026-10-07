"""``hermes skills select``: score the skill library against a task with Jev and show the selection.

The same code path the agent runs per message when ``skills.selection.enabled`` is on, minus the
conversation: the task comes from the command line, stdin, or ``--prompt``, and optional recent
context from ``--context-file``. Skills the CLI agent would hide (platform, toolset and session gates)
are hidden here too. ``--dry-run`` prints the exact request bodies without a key or network.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

def _read_prompt(args: Any) -> Optional[str]:
    positional, option = getattr(args, "prompt", None), getattr(args, "prompt_opt", None)
    if positional and option:
        raise ValueError("give the task once: as an argument or with --prompt, not both")
    text = option if option is not None else positional
    if text == "-":
        text = sys.stdin.read()
    return text.strip() if isinstance(text, str) and text.strip() else None


def _read_context(path: Optional[str]) -> List[Dict[str, str]]:
    """Recent conversation for scoring: a JSON list of ``{"role", "text"|"content"}`` objects, or
    plain text treated as one earlier user message."""
    if not path:
        return []
    raw = Path(path).expanduser().read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        out = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            text = item.get("text", item.get("content"))
            role = str(item.get("role") or "user")
            if isinstance(text, str) and text.strip() and role in ("user", "assistant"):
                out.append({"role": role, "text": text.strip()})  # clipped later, as the agent clips
        return out
    return [{"role": "user", "text": raw}]


def _config_from_args(args: Any):
    from agent.relevance_skills import load_selection_config

    cfg = load_selection_config()
    overrides = {k: v for k, v in (("min_score", args.min_score), ("target_score", args.target_score),
                                   ("token_budget", args.token_budget)) if v is not None}
    return dataclasses.replace(cfg, **overrides)


def _cli_visibility() -> Dict[str, Any]:
    """The tools, toolsets and platform the CLI agent runs with, so skills it hides stay hidden here."""
    import model_tools
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools

    config = load_config()
    toolsets = sorted(_get_platform_tools(config, "cli", include_default_mcp_servers=False))
    tools = {d["function"]["name"] for d in model_tools.get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)}
    # Derived from the tools, exactly as the agent's per-turn selection does.
    available = {model_tools.get_toolset_for_tool(t) for t in tools} - {None, ""}
    # MCP servers' tools are only known once they connect, and the preview never starts them; their toolsets
    # count as available, as they are for the agent once it starts.
    available |= _get_platform_tools(config, "cli") - set(toolsets)
    return {"available_tools": tools, "available_toolsets": available, "platform": "cli"}


def _print_table(report: Any) -> None:
    data = report.to_dict()
    sel = report.selection
    print(f"Model {data['model']} · {data['requests']} request(s) · {data['elapsed_ms']} ms · "
          f"{data['usage']['input_tokens']} input tokens · {data['candidates']} skills scored")
    print(f"min_score {data['min_score']:g} · target {data['target_score']:g} · budget {data['token_budget']} tokens")
    print()
    if not sel.selected:
        print("No skill selected.")
    else:
        print(f"Selected ({len(sel.selected)}; total score {sel.total_score:.1f}, ~{sel.tokens} tokens"
              f"{', target reached' if sel.target_reached else ''}):")
        for d in (d for d in sel.decisions if d.status == "selected"):
            print(f"  {d.score:4.1f}  {d.name}  (~{d.tokens} tokens)")
    passed = [d for d in sel.decisions if d.status not in ("selected", "below_min_score")]
    if passed:
        print("\nQualified but not attached:")
        for d in passed:
            print(f"  {d.score:4.1f}  {d.name}  [{d.status}]")
    nearest = [d for d in sel.decisions if d.status == "below_min_score"][:5]
    if nearest:
        print("\nHighest below min_score:")
        for d in nearest:
            print(f"  {d.score:4.1f}  {d.name}")


def cmd_skills_select(args: Any) -> int:
    from agent.relevance import RelevanceError, api_key, describe_error, load_settings
    from agent.relevance_skills import (
        api_key_hint, build_state, collect_candidates, outbound_requests, recent_conversation, run_selection,
    )
    from agent.relevance_typesafe import TYPESAFE_ENDPOINT

    try:
        request = _read_prompt(args)
        recent = _read_context(getattr(args, "context_file", None))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not request:
        print("error: no task given (pass it as an argument, with --prompt, or '-' to read stdin)", file=sys.stderr)
        return 2
    cfg = _config_from_args(args)
    # The agent sends the last recent_messages user/assistant messages, each cut to 600 characters.
    recent = recent_conversation([{"role": r["role"], "content": r["text"]} for r in recent], limit=cfg.recent_messages)
    settings = load_settings()
    visibility = _cli_visibility()
    if args.dry_run:
        candidates = collect_candidates(**visibility)
        try:
            bodies = outbound_requests(candidates, build_state(request, recent), settings=settings,
                                       need_gate=cfg.need_gate > 0)
        except RelevanceError as exc:
            print(f"error: {describe_error(exc)}", file=sys.stderr)
            return 1
        print(json.dumps({"dry_run": True, "endpoint": TYPESAFE_ENDPOINT, "candidates": len(candidates),
                          "skills": [c.name for c in candidates], "requests": bodies}, indent=2, ensure_ascii=False))
        return 0
    if not api_key():
        print(f"error: TYPESAFE_API_KEY is not set ({api_key_hint()}); use --dry-run to preview offline",
              file=sys.stderr)
        return 1
    try:
        report = run_selection(request, config=cfg, settings=settings, recent=recent, **visibility)
    except RelevanceError as exc:
        print(f"error: skill scoring failed: {describe_error(exc)}", file=sys.stderr)
        return 1
    if args.format == "context":
        if report.selection.context:
            print(report.selection.context)
    elif args.format == "table":
        _print_table(report)
    else:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0
