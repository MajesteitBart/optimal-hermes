"""Relevance decision ledger: append, rotate, read back, summarize, and the off switch."""

from __future__ import annotations

import time

from agent import relevance_ledger as ledger


def test_records_read_back_and_summarize():
    ledger.record("skills", selected=[{"name": "pdf", "score": 8.1}], input_tokens=2_000_000, elapsed_ms=300)
    ledger.record("skills", selected=[], input_tokens=0, elapsed_ms=100)
    ledger.record("write_gate", decision="refused", reason="procedure")
    ledger.record("review_gate", run=False)
    summary = ledger.summarize(ledger.read_entries())
    assert summary["skills"]["outcomes"] == {"attached": 1, "none": 1}
    assert summary["skills"]["most_attached"] == [("pdf", 1)]
    assert summary["skills"]["est_cost_usd"] == round(2_000_000 * ledger.USD_PER_INPUT_TOKEN, 4)
    assert summary["write_gate"]["outcomes"] == {"refused": 1}
    assert summary["review_gate"]["outcomes"] == {"skip": 1}


def test_since_filter_and_rotation(monkeypatch):
    for n in range(3):
        ledger.record("skills", n=n)
    monkeypatch.setattr(ledger, "MAX_BYTES", 10)
    ledger.record("skills", n=3)  # over the cap: the current file becomes .1, the entry starts a new file
    assert ledger.ledger_path().with_suffix(".jsonl.1").exists()
    assert [e["n"] for e in ledger.read_entries()] == [0, 1, 2, 3]  # one predecessor is kept and read
    assert ledger.read_entries(since=time.time() + 60) == []


def test_config_turns_it_off():
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text("relevance:\n  ledger: false\n", encoding="utf-8")
    ledger.record("skills", selected=[])
    assert not ledger.ledger_path().exists()


def test_unwritable_ledger_never_raises(monkeypatch):
    monkeypatch.setattr(ledger, "ledger_path", lambda: (_ for _ in ()).throw(OSError("read-only")))
    ledger.record("skills", selected=[])  # must not raise


def test_a_late_record_never_recreates_a_deleted_profile(tmp_path, monkeypatch):
    """Regression (review round 2): a background thread holding a deleted profile's scope must not
    materialize that profile's home again just to append a ledger line."""
    root = tmp_path / ".hermes"
    (root / "profiles").mkdir(parents=True)
    gone = root / "profiles" / "gone"
    monkeypatch.setenv("HERMES_HOME", str(gone))
    ledger.record("skill_outcome", skill="pdf", followed=0.9)
    assert not gone.exists()
