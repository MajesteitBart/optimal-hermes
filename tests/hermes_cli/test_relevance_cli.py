"""CLI surfaces of relevance scoring: ``hermes skills select|overlap``, ``hermes memory audit``,
``hermes relevance status|report``. Subprocess runs cover the offline paths (dry run, missing
key, status); scoring paths run in-process against the TypeSafe stand-in."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._fixtures.typesafe_fake import skill_scores

REPO = Path(__file__).resolve().parents[2]


def _home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def write_skill(name, description, category="tools"):
    path = _home() / "skills" / category / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nSteps.\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _fresh_caches():
    from tools.skills_tool import clear_skills_cache

    clear_skills_cache()
    yield
    clear_skills_cache()


def _hermes(*argv, input_text=None):
    env = {**os.environ, "HERMES_HOME": str(_home()), "PYTHONIOENCODING": "utf-8"}
    env.pop("TYPESAFE_API_KEY", None)
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", *argv], cwd=REPO, env=env, input=input_text,
                          capture_output=True, text=True, encoding="utf-8", timeout=180)


def test_relevance_ends_a_continued_session_name():
    from hermes_cli.main import _coalesce_session_name_args

    # `hermes -c my relevance status` must not become session "my relevance" + the `status` command.
    assert _coalesce_session_name_args(["-c", "my", "relevance", "status"]) == ["-c", "my", "relevance", "status"]


class TestSkillsSelectSubprocess:
    def test_dry_run_prints_the_requests_without_a_key(self):
        write_skill("pdf-tools", "Merge PDFs.")
        proc = _hermes("skills", "select", "--dry-run", "merge two pdfs")
        assert proc.returncode == 0, proc.stderr
        out = json.loads(proc.stdout)
        assert out["dry_run"] is True and out["skills"] == ["pdf-tools"]
        assert out["requests"][0]["state"]["request"] == "merge two pdfs"

    def test_task_from_stdin(self):
        write_skill("pdf-tools", "Merge PDFs.")
        proc = _hermes("skills", "select", "--dry-run", "-", input_text="from stdin\n")
        assert json.loads(proc.stdout)["requests"][0]["state"]["request"] == "from stdin"

    def test_missing_key_is_a_clear_error(self):
        write_skill("pdf-tools", "Merge PDFs.")
        proc = _hermes("skills", "select", "merge two pdfs")
        assert proc.returncode == 1 and "TYPESAFE_API_KEY" in proc.stderr

    def test_no_task_is_a_usage_error(self):
        proc = _hermes("skills", "select")
        assert proc.returncode == 2

    def test_relevance_status_reports_configuration(self):
        (_home() / "config.yaml").write_text("skills:\n  selection:\n    enabled: true\n", encoding="utf-8")
        proc = _hermes("relevance", "status")
        assert proc.returncode == 0
        assert "selection on" in proc.stdout and "NOT SET" in proc.stdout


def _args(**kw):
    defaults = dict(prompt=None, prompt_opt=None, context_file=None, min_score=None, target_score=None,
                    token_budget=None, format="json", dry_run=False)
    return argparse.Namespace(**{**defaults, **kw})


class TestInProcess:
    def test_select_json_contract(self, fake_systemone, capsys, tmp_path):
        from hermes_cli.skills_select import cmd_skills_select

        write_skill("pdf-tools", "Merge PDFs.")
        write_skill("weather", "Forecasts.")
        fake_systemone.answer = skill_scores({"pdf-tools": 8.0, "weather": 1.0})
        context = tmp_path / "ctx.json"
        context.write_text(json.dumps([{"role": "user", "text": "I have two files"}]), encoding="utf-8")
        assert cmd_skills_select(_args(prompt="merge them", context_file=str(context), target_score=5)) == 0
        out = json.loads(capsys.readouterr().out)
        assert [s["name"] for s in out["selected"]] == ["pdf-tools"]
        assert out["target_score"] == 5 and out["target_reached"] is True
        assert {r["name"] for r in out["ranked"]} == {"pdf-tools", "weather"}
        assert fake_systemone.requests[0]["state"]["recent_conversation"] == [{"role": "user", "text": "I have two files"}]

    def test_context_format_prints_the_attached_block(self, fake_systemone, capsys):
        from hermes_cli.skills_select import cmd_skills_select

        write_skill("pdf-tools", "Merge PDFs.")
        fake_systemone.answer = skill_scores({"pdf-tools": 8.0})
        assert cmd_skills_select(_args(prompt="merge", format="context")) == 0
        assert capsys.readouterr().out.startswith("<selected-skills>")

    def test_memory_audit_flags_entries(self, fake_systemone, capsys):
        from hermes_cli.memory_audit import cmd_memory_audit
        from tools.memory_tool import load_on_disk_store

        store = load_on_disk_store()
        store.add("memory", "Fixed the CI job today.")
        store.add("memory", "Builds run on the ops-1 host.")
        fake_systemone.answer = lambda qid, q, state: (
            {"durable": 0.2, "personal": 0.2, "task_state": 0.95} if "CI job" in state["candidate_memory"]
            else {"durable": 0.95, "personal": 0.9}).get(qid, 0.0)
        assert cmd_memory_audit(argparse.Namespace(target="all", json=True)) == 0
        rows = {r["entry"]: r for r in json.loads(capsys.readouterr().out)["entries"]}
        assert rows["Fixed the CI job today."]["keep"] is False
        assert rows["Builds run on the ops-1 host."]["keep"] is True

    def test_overlap_clusters_pairs(self, fake_systemone, capsys):
        from hermes_cli.skills_overlap import cmd_skills_overlap

        write_skill("pptx-author", "Build PowerPoint decks.", category="office")
        write_skill("powerpoint", "Create and edit PowerPoint decks.", category="office")
        write_skill("weather", "Forecasts.")
        fake_systemone.answer = lambda qid, q, state: 0.9 if {q["instructions"]["skill_a"]["name"],
                                                              q["instructions"]["skill_b"]["name"]} == {"pptx-author", "powerpoint"} else 0.05
        assert cmd_skills_overlap(argparse.Namespace(min=0.7, json=True)) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["clusters"] == [["powerpoint", "pptx-author"]]

    def test_report_summarizes_the_ledger(self, capsys):
        from agent.relevance_ledger import record
        from hermes_cli.relevance_cmd import cmd_relevance

        record("skills", selected=[{"name": "pdf-tools", "score": 8}], input_tokens=1000, elapsed_ms=250)
        assert cmd_relevance(argparse.Namespace(relevance_command="report", days=1, json=True)) == 0
        assert json.loads(capsys.readouterr().out)["features"]["skills"]["decisions"] == 1
