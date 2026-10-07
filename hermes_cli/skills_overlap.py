"""``hermes skills overlap``: report skills that do the same job (read-only)."""

from __future__ import annotations

import json
import sys
from typing import Any


def cmd_skills_overlap(args: Any) -> int:
    from agent.relevance import RelevanceError, api_key, describe_error
    from agent.relevance_skill_overlap import find_overlaps
    from agent.relevance_skills import api_key_hint, collect_candidates

    if not api_key():
        print(f"error: TYPESAFE_API_KEY is not set ({api_key_hint()})", file=sys.stderr)
        return 1
    candidates = collect_candidates()
    try:
        report = find_overlaps(candidates, threshold=args.min)
    except RelevanceError as exc:
        print(f"error: overlap scoring failed: {describe_error(exc)}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"skills": len(candidates), "pairs_asked": report.asked, "identical": report.identical,
                          "clusters": report.clusters, "pairs": report.pairs}, indent=2, ensure_ascii=False))
        return 0
    ev = report.evaluation
    print(f"{len(candidates)} skills · {report.asked} candidate pairs judged · "
          f"{ev.requests if ev else 0} request(s) · {ev.input_tokens if ev else 0} input tokens\n")
    if report.identical:
        print("Identical SKILL.md files:")
        for group in report.identical:
            print(f"  {', '.join(group)}")
        print()
    if not report.clusters:
        print(f"No overlapping skills at threshold {args.min:g}.")
        return 0
    print(f"Overlap clusters (threshold {args.min:g}):")
    for group in report.clusters:
        print(f"  {', '.join(group)}")
    print("\nPairs:")
    for r in report.pairs:
        print(f"  {r['same_job']:.2f} same job · {r['one_covers_other']:.2f} one covers other   {r['a']} ~ {r['b']}")
    return 0
