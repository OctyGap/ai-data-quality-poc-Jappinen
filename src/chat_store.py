# Very simple chat history logging per session.
# We log to .logs/<session>.jsonl and also load it back for context.

import json
import time
from pathlib import Path
from typing import List, Dict


LOG_DIR = Path(".logs")


def _path(session_id: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{session_id}.jsonl"


def append_history(session_id: str, role: str, content: str) -> None:
    rec = {"ts": time.time(), "role": role, "content": content}
    with _path(session_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_history(session_id: str) -> List[Dict[str, str]]:
    p = _path(session_id)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            role = rec.get("role")
            content = rec.get("content", "")
            if role in ("user", "assistant"):
                out.append({"role": role, "content": content})
    return out


def trim_history(messages: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    # Keep the most recent messages only, so context fits.
    kept = []
    total = 0
    for m in reversed(messages):
        total += len(m.get("content", ""))
        kept.append(m)
        if total >= max_chars:
            break
    return list(reversed(kept))
