"""skills.selection on the wire, with a real AIAgent against a mock provider.

The prompt-cache invariant: attached skills ride the current user message (persisted in its
``api_content`` sidecar and replayed byte-identically), the system prompt is identical across
turns and carries no "load anything partially relevant" policy, and a skill already in context
is not attached a second time.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

from hermes_state import SessionDB
from tests._fixtures.typesafe_fake import skill_scores


class _Provider(BaseHTTPRequestHandler):
    captured: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
        type(self).captured.append(req)
        resp = {"id": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"},
                                        "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
        if req.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in ({"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"},
                                                   "finish_reason": None}]},
                          {"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


@pytest.fixture()
def env(fake_systemone):
    from hermes_constants import get_hermes_home
    from tools.skills_tool import clear_skills_cache

    home = get_hermes_home()
    skill = home / "skills" / "docs" / "pdf-tools" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text("---\nname: pdf-tools\ndescription: Merge and split PDF files.\n---\n\n# PDF tools\n\n"
                     "Merge with: pdftk a.pdf b.pdf cat output out.pdf\n", encoding="utf-8")
    (home / "config.yaml").write_text("skills:\n  selection:\n    enabled: true\n", encoding="utf-8")
    clear_skills_cache()
    fake_systemone.answer = skill_scores({"pdf-tools": 8.0})

    _Provider.captured = []
    server = HTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    db = SessionDB(db_path=home / "state.db")  # the agent derives its home (and skills dir) from the DB
    sid = "sess-skills"

    def make_agent():
        from run_agent import AIAgent

        agent = AIAgent(api_key="test-key", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                        provider="openai-compat", model="test-model", max_iterations=4, enabled_toolsets=[],
                        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                        platform="cli", session_db=db, session_id=sid)
        agent.valid_tool_names = {"skill_view", "skills_list", "skill_manage"}
        return agent

    try:
        yield make_agent, db, sid, fake_systemone
    finally:
        server.shutdown()
        db.close()


def _chats():
    return [r for r in _Provider.captured if "messages" in r]


def _users(req):
    return [m for m in req["messages"] if m["role"] == "user"]


def _system(req):
    return next(m["content"] for m in req["messages"] if m["role"] == "system")


def test_attached_skills_ride_the_user_message_and_replay_byte_identically(env):
    make_agent, db, sid, fake = env

    make_agent().run_conversation("please merge these two pdf files", conversation_history=[], task_id="t1")
    first = _chats()[0]
    sent = _users(first)[0]["content"]
    assert sent.startswith("please merge these two pdf files\n\n<selected-skills>")
    assert "pdftk a.pdf b.pdf" in sent
    system_1 = _system(first)
    assert "MUST load" not in system_1 and "<selected-skills>" in system_1 and "pdf-tools" in system_1

    row = [r for r in db.get_messages(sid) if r["role"] == "user"][0]
    assert row["content"] == "please merge these two pdf files" and row["api_content"] == sent

    # Turn 2: fresh agent, history reloaded from the store; the same skill still scores high.
    _Provider.captured = []
    requests_before = len(fake.requests)
    history = db.get_messages_as_conversation(sid)
    make_agent().run_conversation("now split the merged pdf", conversation_history=history, task_id="t2")
    second = _chats()[0]
    assert _users(second)[0]["content"] == sent  # turn 1 replayed byte for byte
    assert _system(second) == system_1  # the system prompt never changed
    assert "<selected-skills>" not in _users(second)[-1]["content"]  # already in context: not attached again
    scored_names = [q["instructions"]["skill"]["name"] for body in fake.requests[requests_before:]
                    for q in body["questions"].values() if q["type"] == "score"]
    assert "pdf-tools" not in scored_names  # excluded before scoring, not just before attaching
