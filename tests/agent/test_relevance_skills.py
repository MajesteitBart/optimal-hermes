"""Per-message skill selection (skills.selection): greedy selection rules, the scoring pipeline
against synthetic skills, context awareness, the per-turn entry point and the prompt index."""

from __future__ import annotations

import json
import types
from pathlib import Path

import httpx
import pytest

import agent.relevance_skills as rs
from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
from tests._fixtures.typesafe_fake import skill_scores


def _home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def write_skill(name, description, *, category="tools", body="Follow these steps.", frontmatter=""):
    path = _home() / "skills" / category / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n{frontmatter}---\n\n# {name}\n\n{body}\n",
                    encoding="utf-8")
    return path


def write_config(text):
    (_home() / "config.yaml").write_text(text, encoding="utf-8")


@pytest.fixture(autouse=True)
def _fresh_skill_caches():
    from tools.skills_tool import clear_skills_cache

    clear_skills_cache()
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    clear_skills_cache()


def _candidate(name, *, content_hash=None):
    return rs.SkillCandidate(name=name, description=f"{name} skill", category="tools", path=Path(name),
                             content_hash=content_hash or name)


def _scored(name, score, **kw):
    return rs.ScoredSkill(candidate=_candidate(name, **kw), score=score, confidence=0.9)


def _render(sizes):
    return lambda item: "x" * sizes.get(item.candidate.name, 40)


class TestSelectSkills:
    def test_nothing_qualifies_attaches_nothing(self):
        sel = rs.select_skills([_scored("a", 5.9), _scored("b", 1)], min_score=6, target_score=18, token_budget=6000,
                               render=_render({}))
        assert sel.context == "" and sel.selected == []
        assert {d.status for d in sel.decisions} == {rs.BELOW_MIN}

    def test_target_is_a_stopping_point_not_a_quota(self):
        sel = rs.select_skills([_scored("a", 9), _scored("b", 9), _scored("c", 8)], min_score=6, target_score=18,
                               token_budget=6000, render=_render({}))
        assert [s.candidate.name for s in sel.selected] == ["a", "b"]
        assert sel.target_reached and [d.status for d in sel.decisions][-1] == rs.TARGET_REACHED

    def test_oversized_skill_is_skipped_and_a_smaller_one_still_fits(self):
        sel = rs.select_skills([_scored("big", 9), _scored("small", 7)], min_score=6, target_score=18,
                               token_budget=200, render=_render({"big": 4000, "small": 100}))
        assert [s.candidate.name for s in sel.selected] == ["small"]
        assert sel.decisions[0].status == rs.OVER_BUDGET and sel.tokens <= 200

    def test_budget_counts_the_wrapper_not_only_the_body(self):
        body_tokens = rs.estimate_tokens_rough("x" * 400)
        sel = rs.select_skills([_scored("a", 9)], min_score=6, target_score=18, token_budget=body_tokens,
                               render=_render({"a": 400}))
        assert sel.selected == [] and sel.decisions[0].status == rs.OVER_BUDGET

    def test_identical_content_is_attached_once(self):
        sel = rs.select_skills([_scored("a", 9, content_hash="h"), _scored("b", 8, content_hash="h")], min_score=6,
                               target_score=18, token_budget=6000, render=_render({}))
        assert [s.candidate.name for s in sel.selected] == ["a"]
        assert sel.decisions[1].status == rs.DUPLICATE

    def test_unloadable_skill_is_reported_and_skipped(self):
        sel = rs.select_skills([_scored("a", 9), _scored("b", 8)], min_score=6, target_score=18, token_budget=6000,
                               render=lambda item: None if item.candidate.name == "a" else "body")
        assert [d.status for d in sel.decisions] == [rs.UNLOADABLE, rs.SELECTED]

    def test_ties_break_by_name_for_stable_results(self):
        sel = rs.select_skills([_scored("zeta", 7), _scored("alpha", 7)], min_score=6, target_score=7,
                               token_budget=6000, render=_render({}))
        assert [s.candidate.name for s in sel.selected] == ["alpha"]


class TestPipeline:
    def _setup(self):
        write_skill("pdf-tools", "Read, merge and split PDF files.", body="Use pdftk to merge.")
        write_skill("deck-maker", "Build slide decks.", body="Use python-pptx.")
        write_skill("weather", "Weather forecasts.", body="Call the weather API.")

    def test_selects_highest_first_and_reports_usage(self, fake_systemone):
        self._setup()
        fake_systemone.answer = skill_scores({"pdf-tools": 8.5, "deck-maker": 6.5, "weather": 1})
        cfg = rs.SelectionConfig(enabled=True, rerank=False)
        report = rs.run_selection("merge these two PDFs", config=cfg)
        assert [s.candidate.name for s in report.selection.selected] == ["pdf-tools", "deck-maker"]
        assert "[Selected skill \"pdf-tools\"" in report.selection.context
        assert "Use pdftk to merge." in report.selection.context  # the full SKILL.md body, not a summary
        assert report.to_dict()["usage"]["input_tokens"] == 100 * report.evaluation.requests

    def test_wide_pass_sends_descriptions_but_never_bodies_or_paths(self, fake_systemone):
        self._setup()
        rs.run_selection("merge PDFs", config=rs.SelectionConfig(enabled=True, rerank=False))
        wire = json.dumps(fake_systemone.requests)
        assert "Read, merge and split PDF files." in wire
        assert "Use pdftk" not in wire and str(_home()) not in wire

    def test_close_read_rescores_only_near_qualifiers_with_an_excerpt(self, fake_systemone):
        self._setup()

        def answer(qid, question, state):
            skill = question["instructions"]["skill"]
            if "instructions_excerpt" in skill:  # close read: pdf-tools turns out not to fit
                return {"pdf-tools": 2.0, "deck-maker": 7.0}[skill["name"]]
            return {"pdf-tools": 8.0, "deck-maker": 4.5, "weather": 1.0}[skill["name"]]

        fake_systemone.answer = answer
        report = rs.run_selection("make slides", config=rs.SelectionConfig(enabled=True, rerank=True))
        close = [q for body in fake_systemone.requests for q in body["questions"].values()
                 if "instructions_excerpt" in q["instructions"]["skill"]]
        assert sorted(q["instructions"]["skill"]["name"] for q in close) == ["deck-maker", "pdf-tools"]
        assert [s.candidate.name for s in report.selection.selected] == ["deck-maker"]
        top = report.selection.decisions[0]
        assert (top.name, top.wide_score) == ("deck-maker", pytest.approx(4.5))

    def test_new_topic_drops_stale_turns_from_the_close_read(self, fake_systemone):
        self._setup()
        fake_systemone.answer = skill_scores({"pdf-tools": 7}, nouls={rs.CONTINUES_ID: 0.1})
        recent = [{"role": "user", "text": "about the weather in Utrecht"}]
        report = rs.run_selection("merge PDFs", config=rs.SelectionConfig(enabled=True), recent=recent)
        wide, close = fake_systemone.requests[0], fake_systemone.requests[-1]
        assert rs.CONTINUES_ID in wide["questions"] and wide["state"]["recent_conversation"] == recent
        assert close["state"]["recent_conversation"] == []
        assert report.continues == pytest.approx(0.1)

    def test_needs_skill_gate_vetoes_attachment(self, fake_systemone):
        self._setup()
        fake_systemone.answer = skill_scores({"pdf-tools": 9}, nouls={"gate_acts": 0.05, "gate_procedure": 0.05,
                                                                      "gate_prose": 0.95})
        report = rs.run_selection("what is a PDF?", config=rs.SelectionConfig(enabled=True, need_gate=0.3))
        assert report.selection.selected == [] and report.signals.needs_skill < 0.3
        assert {d.status for d in report.selection.decisions} == {rs.NOT_NEEDED}
        assert len(fake_systemone.requests) == 1  # no close read for a vetoed request

    def test_disabled_and_condition_hidden_skills_are_not_candidates(self):
        self._setup()
        write_skill("needs-browser", "Browse.", frontmatter="metadata:\n  hermes:\n    requires_tools: [browser_navigate]\n")
        write_config("skills:\n  disabled: [weather]\n")
        names = {c.name for c in rs.collect_candidates(available_tools={"skill_view"}, available_toolsets={"skills"})}
        assert names == {"pdf-tools", "deck-maker"}

    def test_dry_run_payloads_need_no_key_or_network(self, monkeypatch):
        self._setup()
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        bodies = rs.outbound_requests(rs.collect_candidates(), rs.build_state("merge PDFs"),
                                      settings=rs.load_settings({}))
        sent = [q["instructions"]["skill"]["name"] for b in bodies for q in b["questions"].values()
                if q["type"] == "score"]
        assert sorted(sent) == ["deck-maker", "pdf-tools", "weather"]


class TestContextAwareness:
    def test_detects_selected_slash_and_viewed_skills(self):
        call = {"id": "c1", "type": "function", "function": {"name": "skill_view", "arguments": '{"name": "viewed"}'}}
        pruned = {"id": "c2", "type": "function", "function": {"name": "skill_view", "arguments": '{"name": "pruned"}'}}
        messages = [
            {"role": "user", "content": "hi", "api_content": 'hi\n\n<selected-skills>\n[Selected skill "attached" · relevance 7.0/9]'},
            {"role": "user", "content": '[IMPORTANT: The user has invoked the "slashed" skill, indicating ...'},
            {"role": "assistant", "content": "", "tool_calls": [call, pruned]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"success": True, "name": "viewed", "content": "..."})},
            {"role": "tool", "tool_call_id": "c2", "content": "[tool result pruned]"},
        ]
        assert rs.skills_in_context(messages) == {"attached", "slashed", "viewed"}

    def test_recent_conversation_uses_durable_text_only(self):
        messages = [
            {"role": "user", "content": "first ask", "api_content": "first ask\n\n<memory-context>secret</memory-context>"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
            {"role": "tool", "tool_call_id": "x", "content": "tool output"},
            {"role": "assistant", "content": "done it"},
        ]
        assert rs.recent_conversation(messages, limit=4) == [
            {"role": "user", "text": "first ask"}, {"role": "assistant", "text": "done it"}]


def _agent(**kw):
    statuses = []
    agent = types.SimpleNamespace(valid_tool_names={"skill_view", "skills_list", "skill_manage"}, platform="cli",
                                  session_id="sess-1", _cached_system_prompt="", _emit_status=statuses.append,
                                  _emit_warning=statuses.append, **kw)
    return agent, statuses


class TestTurnEntryPoint:
    def test_off_by_default(self, fake_systemone):
        write_skill("pdf-tools", "PDFs.")
        agent, _ = _agent()
        assert rs.build_turn_skill_context(agent, user_message="merge PDFs", messages=[{"role": "user"}],
                                           current_turn_user_idx=0) == ""
        assert fake_systemone.requests == []

    def test_attaches_reports_and_records(self, fake_systemone):
        write_skill("pdf-tools", "PDFs.")
        write_config("skills:\n  selection:\n    enabled: true\n")
        fake_systemone.answer = skill_scores({"pdf-tools": 8})
        agent, statuses = _agent()
        block = rs.build_turn_skill_context(agent, user_message="merge these PDFs", task_id="t1",
                                            messages=[{"role": "user", "content": "merge these PDFs"}],
                                            current_turn_user_idx=0)
        assert block.startswith(rs.SELECTED_SKILLS_OPEN) and "pdf-tools" in block
        assert statuses == ["🧩 Skills attached: pdf-tools (8.0)"]
        from agent.relevance_ledger import read_entries
        from tools.skills_tool_dedup import _check_skill_view_dedup

        entry = read_entries()[-1]
        assert entry["feature"] == "skills" and entry["selected"] == [{"name": "pdf-tools", "score": 8.0}]
        assert "merge these PDFs" not in json.dumps(entry)  # the ledger never stores message text
        assert _check_skill_view_dedup("t1", "pdf-tools", None)  # a follow-up skill_view gets a stub

    def test_trivial_and_explicit_skill_turns_do_not_score(self, fake_systemone):
        write_skill("pdf-tools", "PDFs.")
        write_config("skills:\n  selection:\n    enabled: true\n")
        agent, _ = _agent()
        for text in ("thanks!", '[IMPORTANT: The user has invoked the "pdf-tools" skill, indicating ...'):
            assert rs.build_turn_skill_context(agent, user_message=text, messages=[], current_turn_user_idx=0) == ""
        assert fake_systemone.requests == []

    def test_failure_attaches_nothing_and_warns_once(self, fake_systemone):
        write_skill("pdf-tools", "PDFs.")
        write_config("skills:\n  selection:\n    enabled: true\nrelevance:\n  max_retries: 0\n")
        fake_systemone.replies = [httpx.Response(500, text="SECRET-BODY"), httpx.Response(500, text="SECRET-BODY")]
        agent, statuses = _agent()
        for _ in range(2):
            assert rs.build_turn_skill_context(agent, user_message="merge PDFs", messages=[],
                                               current_turn_user_idx=0) == ""
        assert len(statuses) == 1 and statuses[0].startswith("⚠ Skill selection unavailable (http_500)")
        assert "SECRET-BODY" not in statuses[0]


class TestPromptIndex:
    def _build(self, mode):
        clear_skills_system_prompt_cache(clear_snapshot=True)
        return build_skills_system_prompt(available_tools={"skill_view", "skills_list", "skill_manage"},
                                          selection_index=mode)

    def test_selection_modes_drop_the_load_everything_policy(self):
        write_skill("pdf-tools", "Read and merge PDFs.")
        write_skill("hermes-agent", "Configure Hermes.", category="meta")
        default = self._build(None)
        assert "MUST load" in default
        names, full, none = self._build("names"), self._build("full"), self._build("none")
        for prompt in (names, full, none):
            assert "MUST load" not in prompt and "<selected-skills>" in prompt
        assert "pdf-tools" in names and "Read and merge PDFs." not in names
        assert "Read and merge PDFs." in full
        assert "pdf-tools" not in none and "skills_list" in none

    def test_hermes_agent_help_guidance_survives_names_mode(self):
        from agent.system_prompt import _HERMES_AGENT_SKILL_LISTED

        write_skill("hermes-agent", "Configure Hermes.", category="meta")
        assert _HERMES_AGENT_SKILL_LISTED.search(self._build("names"))
        assert _HERMES_AGENT_SKILL_LISTED.search(self._build(None))


class TestOutcomes:
    def _agent_with_attached(self):
        agent, _ = _agent()
        agent._turn_attached_skills = [("pdf-tools", "Merge PDFs."), ("weather", "Forecasts.")]
        return agent

    def test_records_whether_the_reply_followed_each_skill(self, fake_systemone, monkeypatch):
        write_config("skills:\n  selection:\n    enabled: true\n    track_outcomes: true\n")
        fake_systemone.answer = lambda qid, q, state: 0.9 if q["instructions"]["skill"]["name"] == "pdf-tools" else 0.1
        started = []
        monkeypatch.setattr("agent.memory_provider.spawn_context_thread",
                            lambda target, name, args=(), **kw: types.SimpleNamespace(start=lambda: started.append(target(*args))))
        agent = self._agent_with_attached()
        rs.track_turn_outcomes(agent, "merge the PDFs", "Ran pdftk a.pdf b.pdf cat output out.pdf")
        from agent.relevance_ledger import read_entries, summarize

        outcomes = {e["skill"]: e["followed"] for e in read_entries() if e["feature"] == "skill_outcome"}
        assert outcomes == {"pdf-tools": 0.9, "weather": 0.1}
        assert summarize(read_entries())["skill_outcome"]["per_skill"][0][0] == "weather"  # least followed first
        assert fake_systemone.requests[0]["state"]["reply"].startswith("Ran pdftk")
        assert agent._turn_attached_skills == []

    def test_off_by_default_and_never_sends_the_reply(self, fake_systemone):
        write_config("skills:\n  selection:\n    enabled: true\n")
        agent = self._agent_with_attached()
        rs.track_turn_outcomes(agent, "merge", "done")
        assert fake_systemone.requests == [] and agent._turn_attached_skills == []


class TestInertAttachment:
    """Regression (review): selection read skills through skill_view, which prompts for secrets and installs
    `deps`, and preprocessing could run inline shell, before the budget check. Attaching must be inert."""

    def test_attaching_never_activates_or_runs_shell(self, fake_systemone, monkeypatch):
        import pm

        write_skill("media-tool", "Convert media files.", body="Run !`echo should-not-run` first.",
                    frontmatter="deps: [ffmpeg]\nrequired_environment_variables:\n  - name: MEDIA_TOKEN\n    prompt: token\n")
        write_config("skills:\n  inline_shell: true\n")
        monkeypatch.setattr(pm, "ensure", lambda *a, **k: pytest.fail("pm.ensure ran during selection"))
        monkeypatch.setattr("agent.skill_preprocessing.expand_inline_shell",
                            lambda *a, **k: pytest.fail("inline shell ran during selection"))
        monkeypatch.setattr("tools.skills_tool._secret_capture_callback",
                            lambda *a, **k: pytest.fail("secret prompt during selection"))
        fake_systemone.answer = skill_scores({"media-tool": 8})
        report = rs.run_selection("convert this video", config=rs.SelectionConfig(enabled=True, rerank=False,
                                                                                  token_budget=0))
        assert report.selection.selected == []  # over budget: nothing attached, and nothing activated above
        report = rs.run_selection("convert this video", config=rs.SelectionConfig(enabled=True, rerank=False))
        block = report.selection.context
        assert "!`echo should-not-run`" in block and 'skill_view(name="media-tool")' in block

    def test_setup_skills_are_not_registered_for_the_repeat_view_stub(self, fake_systemone):
        from tools.skills_tool_dedup import _check_skill_view_dedup

        write_skill("media-tool", "Convert media.", frontmatter="deps: [ffmpeg]\n")
        write_config("skills:\n  selection:\n    enabled: true\n")
        fake_systemone.answer = skill_scores({"media-tool": 8})
        agent, _ = _agent()
        assert rs.build_turn_skill_context(agent, user_message="convert this video", task_id="t9",
                                           messages=[], current_turn_user_idx=0)
        assert _check_skill_view_dedup("t9", "media-tool", None) is None  # skill_view still activates it


class TestReviewRegressionsRound3:
    """Regressions from the branch review (round 3)."""

    def test_a_stalled_endpoint_cannot_hold_the_turn_past_its_budget(self, fake_systemone):
        import time

        write_skill("pdf-tools", "PDFs.")
        write_config("skills:\n  selection:\n    enabled: true\nrelevance:\n  turn_budget_seconds: 1\n"
                     "  max_retries: 5\n")

        def stalled(request):
            time.sleep(0.4)
            raise httpx.ReadTimeout("slow")

        fake_systemone.handle = stalled
        agent, statuses = _agent()
        started = time.monotonic()
        assert rs.build_turn_skill_context(agent, user_message="merge PDFs", messages=[],
                                           current_turn_user_idx=0) == ""
        assert time.monotonic() - started < 3.0  # a 1 s budget, not 6 attempts x 8 s
        assert statuses == ["⚠ Skill selection unavailable (timeout); continuing without attached skills"]

    def test_settings_come_from_the_agents_own_home(self, tmp_path):
        other = tmp_path / "other-home"
        other.mkdir()
        (other / "config.yaml").write_text("skills:\n  selection:\n    enabled: true\n", encoding="utf-8")
        write_config("skills:\n  selection:\n    enabled: false\n")  # the ambient (launch) home
        agent = types.SimpleNamespace(_session_db=types.SimpleNamespace(db_path=str(other / "state.db")))
        assert rs.agent_selection_config(agent).enabled is True

    def test_delegated_children_do_not_select(self, fake_systemone):
        write_skill("pdf-tools", "PDFs.")
        write_config("skills:\n  selection:\n    enabled: true\n")
        agent, _ = _agent(_delegate_depth=1)
        assert rs.build_turn_skill_context(agent, user_message="merge PDFs", messages=[],
                                           current_turn_user_idx=0) == ""
        assert fake_systemone.requests == []

    def test_index_none_keeps_the_hermes_help_pointer_and_rename_note(self):
        from agent.prompt_builder import _render_selection_index
        from agent.system_prompt import _HERMES_AGENT_SKILL_LISTED

        index = _render_selection_index({"meta": [("hermes-agent", "Configure Hermes.")]}, {}, {"skills_list"}, "none",
                                        unloadable=["dup-skill"])
        assert _HERMES_AGENT_SKILL_LISTED.search(index) and "dup-skill" in index
