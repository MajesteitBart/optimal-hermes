"""Memory gate evals: write gate, review gate and recall filter on labeled synthetic data.

Scores are fetched once (cached in results/typesafe_cache.json) and the decision rules are
re-applied offline, so threshold sweeps cost nothing after the first run.

Usage (needs TYPESAFE_API_KEY):

    python evals/context_relevance/memory_gates_eval.py [--only write,review,recall]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any, Dict, List

from common import RESULTS, CachingTransport, load_json

import agent.relevance_memory as rm
from agent.relevance import NoulAnswer, RelevanceError, load_settings

SETTINGS = load_settings({})


def _pool_map(fn, items, workers=6):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def write_gate(transport: CachingTransport) -> Dict[str, Any]:
    cases = load_json("memory_candidates.json")

    def run(case):
        try:
            verdict = rm.assess_writes([(case["content"], case["target"], case.get("existing") or [])],
                                       user_message=case.get("user_message") or "", settings=SETTINGS,
                                       transport=transport)[0]
        except RelevanceError as exc:
            return {"id": case["id"], "error": str(exc)}
        # Routed to replace (the candidate supersedes an entry) keeps the information: count it as keep.
        return {"id": case["id"], "label": case["label"], "kind": case["kind"], "scores": verdict.scores,
                "keep": verdict.keep or verdict.supersedes is not None, "replace": verdict.supersedes is not None,
                "overlap": rm.overlapping_entries(case["content"], case.get("existing") or []),
                "target": case["target"], "reason": verdict.reason}

    rows = _pool_map(run, cases)
    ok = [r for r in rows if "error" not in r]

    def evaluate(rows_):
        correct = [(r["keep"] == (r["label"] == "keep")) for r in rows_]
        kept_ok = [r["label"] == "keep" for r in rows_ if r["keep"]]
        rejects = [r for r in rows_ if r["label"] == "reject"]
        by_kind = {k: round(mean((r["keep"] == (r["label"] == "keep")) for r in rows_ if r["kind"] == k), 3)
                   for k in sorted({r["kind"] for r in rows_})}
        return {"accuracy": round(mean(correct), 3) if correct else None,
                "keep_precision": round(mean(kept_ok), 3) if kept_ok else None,
                "reject_recall": round(mean(not r["keep"] for r in rejects), 3) if rejects else None,
                "lost_keeps": [r["id"] for r in rows_ if r["label"] == "keep" and not r["keep"]],
                "by_kind": by_kind}

    report = {"n": len(rows), "errors": len(rows) - len(ok), "default": evaluate(ok)}
    sweep = {}
    defaults = rm.LASTING_VALUE_MIN, rm.TASK_STATE_MIN
    for value_min in (0.45, 0.55, 0.6, 0.65):
        for task_min in (0.6, 0.7, 0.8):
            rm.LASTING_VALUE_MIN, rm.TASK_STATE_MIN = value_min, task_min
            rejudged = []
            for r in ok:
                v = rm.judge_write({k: NoulAnswer(x) for k, x in r["scores"].items()}, target=r["target"],
                                   overlap=r["overlap"])
                rejudged.append({**r, "keep": v.keep or v.supersedes is not None})
            sweep[f"value>={value_min} task<{task_min}"] = evaluate(rejudged)["accuracy"]
    rm.LASTING_VALUE_MIN, rm.TASK_STATE_MIN = defaults
    report["sweep_accuracy"] = sweep
    report["rows"] = rows
    return report


def review_gate(transport: CachingTransport) -> Dict[str, Any]:
    cases = load_json("review_conversations.json")

    def run(case):
        try:
            messages = [{"role": m["role"], "content": m["text"]} for m in case["messages"]]
            verdict = rm.review_worthwhile(messages, settings=SETTINGS, transport=transport)
        except RelevanceError as exc:
            return {"id": case["id"], "error": str(exc)}
        return {"id": case["id"], "label": bool(case["label"]), "strongest": verdict.strongest, "scores": verdict.scores}

    rows = _pool_map(run, cases)
    ok = [r for r in rows if "error" not in r]
    sweep = {}
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        decisions = [(r["strongest"] >= threshold) == r["label"] for r in ok]
        missed = [r["id"] for r in ok if r["label"] and r["strongest"] < threshold]
        skipped = mean(r["strongest"] < threshold for r in ok) if ok else 0
        sweep[str(threshold)] = {"accuracy": round(mean(decisions), 3) if decisions else None,
                                 "missed_durable": missed, "reviews_skipped_share": round(skipped, 3)}
    return {"n": len(rows), "errors": len(rows) - len(ok), "default_threshold": rm.REVIEW_MIN, "sweep": sweep,
            "rows": rows}


def recall_filter(transport: CachingTransport) -> Dict[str, Any]:
    cases = load_json("recall_cases.json")

    def run(case):
        try:
            result = rm.filter_recall(case["recall"], message=case["message"], settings=SETTINGS, transport=transport)
        except RelevanceError as exc:
            return {"id": case["id"], "error": str(exc)}
        return {"id": case["id"], "relevant": case["relevant"], "items": result.items}

    rows = _pool_map(run, cases)
    ok = [r for r in rows if "error" not in r]

    def evaluate(judge):
        precision, recall, injections_kept = [], [], 0
        for r in ok:
            gold = {t.strip() for t in r["relevant"]}
            kept = {it["text"].lstrip("-*+ ").strip() for it in r["items"] if judge(it["scores"])}
            if kept:
                precision.append(len(kept & gold) / len(kept))
            if gold:
                recall.append(len(kept & gold) / len(gold))
            injections_kept += sum(1 for it in r["items"] if judge(it["scores"]) and it["scores"]["injection"] >= 0.5
                                   and it["text"].lstrip("-*+ ").strip() not in gold)
        return {"precision": round(mean(precision), 3) if precision else None,
                "recall": round(mean(recall), 3) if recall else None, "suspect_injections_kept": injections_kept}

    sweep = {}
    default_rel = rm.RECALL_RELEVANT_MIN
    for rel in (0.2, 0.3, 0.4, 0.5, 0.6):
        rm.RECALL_RELEVANT_MIN = rel
        sweep[f"relevant>={rel}"] = evaluate(rm.judge_recall_item)
    rm.RECALL_RELEVANT_MIN = default_rel
    return {"n": len(rows), "errors": len(rows) - len(ok), "default": evaluate(rm.judge_recall_item),
            "sweep": sweep, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="write,review,recall")
    args = parser.parse_args()
    transport = CachingTransport()
    runners = {"write": write_gate, "review": review_gate, "recall": recall_filter}
    out: Dict[str, Any] = {}
    for name in args.only.split(","):
        started = time.monotonic()
        out[name] = runners[name](transport)
        out[name]["wall_s"] = round(time.monotonic() - started, 1)
        transport.save()
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "rows"}, ensure_ascii=False))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"memory_gates_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"\nwrote {path} (cache hits {transport.hits}, misses {transport.misses})")


if __name__ == "__main__":
    main()
