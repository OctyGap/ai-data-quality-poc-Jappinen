# CLI entry + handle_prompt() used by web.py.
# Loads one file once into a global DataFrame.

import argparse
import os
import uuid

import pandas as pd
from dotenv import load_dotenv

from src.llm import run_agent
from src.chat_store import load_history, append_history, trim_history


_DF = None
_SCHEMA = None

def load_file_once(path: str) -> None:
    global _DF, _SCHEMA
    if _DF is not None:
        return

    p = path.lower()
    if p.endswith(".csv"):
        _DF = pd.read_csv(path)
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        _DF = pd.read_excel(path)
    else:
        raise ValueError("Use .csv or .xlsx/.xls")

    _SCHEMA = [{"name": str(c), "dtype": str(_DF[c].dtype)} for c in _DF.columns]

def handle_prompt(prompt: str, session_id: str, file_path: str) -> str:
    load_file_once(file_path)

    history = load_history(session_id)
    history.append({"role": "user", "content": prompt})
    append_history(session_id, "user", prompt)

    context = trim_history(history, max_chars=200)

    answer, new_msgs, rows_for_user = run_agent(df=_DF, schema=_SCHEMA, messages=context, session_id=session_id)

    for m in new_msgs:
        append_history(session_id, m["role"], m["content"])

    # Show rows to the user (CLI) without sending them to LLM.
    if rows_for_user is not None and len(rows_for_user) > 0:
        print("\n--- rows (not shared with LLM) ---")
        try:
            print(rows_for_user.to_string(index=False))
        except Exception:
            print(rows_for_user.head(10))
        print("--- end ---\n")

    return answer

def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY (set env var or put in .env).")

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to one CSV/XLSX file")
    ap.add_argument("--session", default=None, help="Session id")
    ap.add_argument("--interactive", action="store_true", help="REPL chat")
    ap.add_argument("prompt", nargs="?", help='Prompt in quotes')
    args = ap.parse_args()

    sid = args.session or str(uuid.uuid4())
    print(f"session: {sid}")

    if args.interactive:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            print(handle_prompt(q, sid, args.file))
        return

    if not args.prompt:
        ap.print_help()
        return

    print(handle_prompt(args.prompt, sid, args.file))

if __name__ == "__main__":
    main()
