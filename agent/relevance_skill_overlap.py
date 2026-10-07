"""Skill library hygiene: find skills that do the same job.

Candidate pairs are chosen in code (same category, or at least two shared longer words in name and
description); Jev judges each pair with two Noul questions; pairs above the threshold are joined
into clusters. Byte-identical SKILL.md files are reported without asking. Read-only: merging or
archiving stays a human decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.relevance import Evaluation, NoulAnswer, NoulQuestion, RelevanceSettings, evaluate, load_settings
from agent.relevance_skills import SkillCandidate, skill_excerpt

MAX_PAIRS = 800
EXCERPT_CHARS = 600
_WORD_RE = re.compile(r"[a-z0-9]{4,}")
_STOPWORDS = frozenset({"with", "from", "that", "this", "your", "into", "using", "via", "when", "skill",
                        "skills", "files", "file", "tool", "tools", "data", "user", "users", "agent", "hermes"})
STATE = "Overlap review of an assistant's skill library: decide whether two skills do the same job."
SAME_JOB = ("Do `skill_a` and `skill_b` do the same job, so that a user would only ever need one of them for the "
            "tasks they cover?")
CONTAINS = "Does one of `skill_a` and `skill_b` cover everything the other one does?"


def _words(candidate: SkillCandidate) -> set:
    text = f"{candidate.name.replace('-', ' ')} {candidate.description}".lower()
    return {w for w in _WORD_RE.findall(text) if w not in _STOPWORDS}


def candidate_pairs(candidates: Sequence[SkillCandidate], limit: int = MAX_PAIRS) -> List[Tuple[int, int]]:
    """Index pairs worth asking about, most shared words first, capped at ``limit``."""
    words = [_words(c) for c in candidates]
    scored = []
    for i, j in combinations(range(len(candidates)), 2):
        if candidates[i].content_hash == candidates[j].content_hash:
            continue  # identical files are reported without asking
        shared = len(words[i] & words[j])
        if candidates[i].category == candidates[j].category or shared >= 2:
            scored.append((shared, i, j))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(i, j) for _, i, j in scored[:limit]]


@dataclass
class OverlapReport:
    clusters: List[List[str]]
    pairs: List[Dict[str, Any]]  # every pair above the threshold, highest first
    identical: List[List[str]]
    asked: int
    evaluation: Optional[Evaluation] = field(default=None, repr=False)


def _cluster(names: Sequence[str], links: Sequence[Tuple[str, str]]) -> List[List[str]]:
    parent = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b in links:
        parent[find(a)] = find(b)
    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), g))


def find_overlaps(
    candidates: Sequence[SkillCandidate], *, threshold: float = 0.7, settings: Optional[RelevanceSettings] = None,
    key: Optional[str] = None, transport: Any = None,
) -> OverlapReport:
    """Raises :class:`agent.relevance.RelevanceError` when scoring fails."""
    by_hash: Dict[str, List[str]] = {}
    for c in candidates:
        by_hash.setdefault(c.content_hash, []).append(c.name)
    identical = sorted(sorted(names) for names in by_hash.values() if len(names) > 1)
    pairs = candidate_pairs(candidates)
    info = [{"name": c.name, "category": c.category, "description": c.description,
             "instructions_excerpt": skill_excerpt(c.path, EXCERPT_CHARS)} for c in candidates]
    questions: Dict[str, NoulQuestion] = {}
    for n, (i, j) in enumerate(pairs):
        questions[f"same{n}"] = NoulQuestion({"skill_a": info[i], "skill_b": info[j], "question": SAME_JOB})
        questions[f"contains{n}"] = NoulQuestion({"skill_a": info[i], "skill_b": info[j], "question": CONTAINS})
    evaluation = evaluate(STATE, questions, settings=settings or load_settings(), key=key, transport=transport)

    def p(qid: str) -> float:
        answer = evaluation.answers.get(qid)
        return answer.value if isinstance(answer, NoulAnswer) else 0.0

    above = []
    for n, (i, j) in enumerate(pairs):
        same, contains = p(f"same{n}"), p(f"contains{n}")
        if max(same, contains) >= threshold:
            above.append({"a": candidates[i].name, "b": candidates[j].name, "same_job": round(same, 3),
                          "one_covers_other": round(contains, 3)})
    above.sort(key=lambda r: -max(r["same_job"], r["one_covers_other"]))
    links = [(r["a"], r["b"]) for r in above] + [(g[0], other) for g in identical for other in g[1:]]
    clusters = _cluster([c.name for c in candidates], links)
    return OverlapReport(clusters=clusters, pairs=above, identical=identical, asked=len(pairs), evaluation=evaluation)
