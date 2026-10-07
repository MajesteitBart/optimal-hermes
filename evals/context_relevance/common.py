"""Shared plumbing for the context-relevance evals.

- ``CachingTransport``: an httpx transport that answers repeated TypeSafe requests from a JSON
  cache keyed by the exact request body, so re-running a report (or a threshold sweep that
  reuses scores) costs nothing. A cache miss goes to the real API.
- ``isolated_library_home()``: a throwaway HERMES_HOME whose skill library is the repo's public
  bundled + optional skills, so evals never read (or send) a personal skill library.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterator

import httpx

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = Path(__file__).resolve().parent / "data"
RESULTS = Path(__file__).resolve().parent / "results"
CACHE_FILE = RESULTS / "typesafe_cache.json"


class CachingTransport(httpx.BaseTransport):
    """Replays 200 responses for byte-identical request bodies; forwards misses."""

    def __init__(self, cache_file: Path = CACHE_FILE) -> None:
        self._file = cache_file
        self._lock = threading.Lock()
        self._inner = httpx.HTTPTransport()
        self._cache: Dict[str, Any] = {}
        if cache_file.exists():
            self._cache = json.loads(cache_file.read_text(encoding="utf-8"))
        self.hits = 0
        self.misses = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.read()
        key = hashlib.sha256(body).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return httpx.Response(200, json=cached, request=request)
        response = self._inner.handle_request(request)
        response.read()
        if response.status_code == 200:
            with self._lock:
                self._cache[key] = response.json()
        self.misses += 1
        return response

    def save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = json.dumps(self._cache, ensure_ascii=False)
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8", newline="\n")
        tmp.replace(self._file)


@contextlib.contextmanager
def isolated_library_home(*, include_optional: bool = True) -> Iterator[Path]:
    """A temp HERMES_HOME whose skills are the repo's ``skills/`` (+ ``optional-skills/``)."""
    home = Path(tempfile.mkdtemp(prefix="relevance-eval-"))
    dirs = [str(REPO / "skills")] + ([str(REPO / "optional-skills")] if include_optional else [])
    (home / "skills").mkdir()
    (home / "config.yaml").write_text(
        "skills:\n  external_dirs:\n" + "".join(f"    - {d}\n" for d in dirs), encoding="utf-8")
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(home)
    try:
        yield home
    finally:
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous
        shutil.rmtree(home, ignore_errors=True)


def load_json(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[index])
