# This file runs the "agent loop":
# 1) send SYSTEM + chat to the LLM
# 2) LLM returns JSON with an action (SQL/PROFILE/etc.)
# 3) we execute the action locally (safe)
# 4) repeat until FINAL / ASK / EXPORT

import json
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from src.sql_engine import run_sql
from src.profiling import profile_df


# folders for logs + exports
LLM_LOG_DIR = Path(".logs")
LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"
MAX_STEPS = 10

ALLOWED_ACTIONS = {"SCHEMA", "PROFILE", "SEARCH", "SQL", "EXPORT", "ASK", "FINAL"}


# System prompt for the model:
# - return JSON only
# - choose an action
# - do not guess
SYSTEM = """You are a careful assistant over ONE table named `data`.
You MUST compute answers using tools (SQL/SEARCH/SCHEMA/EXPORT/PROFILE). Never guess.
Return ONLY valid JSON. No markdown. No extra text.



Actions:
- "SCHEMA": request schema (columns + types)
- "SEARCH": find candidate values to disambiguate a person/entity
- "SQL": provide a SELECT query to compute the answer
- "EXPORT": provide a SELECT query to export an XLSX file
- "ASK": ask a clarifying question (include a short reason)
- "FINAL": final answer to the user
- "PROFILE": compute profiling WITHOUT SQL

Profiling (PROFILE) kinds supported by profile_df(kind):
- "full" (or "all"): full safe summary
- "missing": missing counts + pct per column (or one column if column is set)
- "outliers": IQR outliers for numeric columns (or one column)
- "duplicates": duplicate rows + duplicate keys per column
- "parse": parse_quality checks (mixed datetime/numeric parsing failures)
- "categorical": categorical summary (masked top values)
- "consistency": consistency checks (negative numeric values, constant columns)

Rules:
- SQL must be SELECT only and single statement.
- If returning rows, include LIMIT 50.
- Use PROFILE for missing/duplicates/outliers/data profiling.
- For EXPORT: do NOT use SELECT *.

Output rules:
- message must be short string (<=2000 chars).
- never invent values.
- FINAL can use template/fill to render from last SQL result.

JSON schema:
{
  "action": "SCHEMA|PROFILE|SEARCH|SQL|EXPORT|ASK|FINAL",
  "sql": "string|null",
  "pattern": "string|null",
  "column": "string|null",
  "profile_kind": "string|null",
  "filename_prefix": "string|null",
  "message": "string",
  "template": "string|null",
  "fill": "object|null",
  "reason": "string|null"
}

When you need values from a SQL result but values are not shown to you:
- If SQL returns a single row with multiple columns, set FINAL like:
  {
    "action":"FINAL",
    "template":"Highest: [highest], Lowest: [lowest], Average: [average].",
    "fill":{"source":"sql_first_row"},
    "message":""
  }
Use [column] placeholders for SQL-first-row fills.
Do NOT put placeholders in message.

If the last tool result came from PROFILE, do NOT use template/fill. Put the final answer directly in "message".
"""


def _sanitize_filename_prefix(prefix):
    # make filename safe 
    p = (prefix or "").strip().lower()
    p = re.sub(r"[^a-z0-9_\-]+", "_", p).strip("_")
    return p[:40] or "export"


def _validate_sql_rules(sql, for_export=False):
    # extra simple rules before we even run sql_engine
    s = (sql or "").strip()
    if not s:
        return "missing sql"

    # allow trailing ; but not multiple statements
    s = s.rstrip(";").strip()
    if ";" in s:
        return "single statement only"

    if not s.lower().startswith("select"):
        return "SELECT only"

    # export must not use SELECT *
    if for_export and re.search(r"(?is)\bselect\s+\*\b", s):
        return "EXPORT queries must not use SELECT *"

    # if not aggregate, force LIMIT so we never dump huge tables
    looks_agg = bool(re.search(r"(?is)\b(count|sum|avg|min|max)\s*\(", s))
    has_limit = bool(re.search(r"(?is)\blimit\b", s))
    if (not for_export) and (not looks_agg) and (not has_limit):
        return "Non-aggregate queries must include LIMIT (use LIMIT 50)"

    return None


def _log_raw_llm(session_id, step, prompt_messages, raw_text):
    # log one json line per step:
    # - date (human readable)
    # - ts (unix)
    # - step number
    # - raw prompt (the messages we sent)
    # - raw answer (what model returned)

    rec = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "ts": time.time(),
        "session_id": session_id,
        "step": step,
        "raw_prompt": prompt_messages,
        "raw_answer": raw_text if len(raw_text) <= 2000 else raw_text[:2000] + "…",
    }

    p = LLM_LOG_DIR / f"{session_id}.llm.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _chat(client, messages, session_id, step):
    # call model, parse JSON, retry once if invalid JSON
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=2000,
    )
    txt = (resp.choices[0].message.content or "").strip()

    try:
        _log_raw_llm(session_id, step, messages, txt)
    except Exception:
        pass

    if not txt:
        raise ValueError("LLM returned empty content")

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        retry_msgs = messages + [{"role": "user", "content": "Return ONLY valid JSON. No markdown, no extra text."}]
        resp2 = client.chat.completions.create(
            model=MODEL,
            messages=retry_msgs,
            temperature=0,
            max_tokens=2000,
        )
        txt2 = (resp2.choices[0].message.content or "").strip()
        if not txt2:
            raise ValueError("LLM returned empty content (retry)")
        return json.loads(txt2)


def _export_xlsx(df, prefix):
    # write xlsx and return filename
    ts = int(time.time())
    safe_prefix = _sanitize_filename_prefix(prefix)
    name = f"{safe_prefix}_{ts}.xlsx"
    df.to_excel(EXPORT_DIR / name, index=False)
    return name

def _fill_bracket_placeholders_from_row(text, row_dict):
    # Replace [Column Name] with value from row_dict
    if not isinstance(text, str) or not text or not isinstance(row_dict, dict):
        return text

    def repl(m):
        key = m.group(0)[1:-1].strip()
        v = row_dict.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            return ""
        return str(v)

    return re.sub(r"\[[^\]]+\]", repl, text)


def _format_first_row(template, df):
    # fill placeholders from the first row
    if df is None or df.empty:
        return "I ran the query, but it returned no rows."

    row = df.iloc[0].to_dict()

    # Preferred: [Column Name] placeholders (works with spaces/punctuation)
    if isinstance(template, str) and "[" in template and "]" in template:
        return _fill_bracket_placeholders_from_row(template, row)

    # Fallback: {col} placeholders (only works for safe column names)
    safe = {}
    for k, v in row.items():
        safe[str(k)] = "" if pd.isna(v) else str(v)

    try:
        return template.format(**safe)
    except Exception:
        # last-resort: try bracket replacement anyway
        return _fill_bracket_placeholders_from_row(template, row)


def _fill_bracket_placeholders(text, scalar):
    # Replace things like [Country Key] or [something] with the actual scalar result
    if scalar is None or not isinstance(text, str) or not text:
        return text
    return re.sub(r"\[[^\]]+\]", str(scalar), text)


def _format_scalar(template, scalar):
    # fill {value} from a scalar result
    try:
        return template.format(value=scalar)
    except Exception:
        return str(scalar)


def _format_rows(template, df, fields, max_items=50):
    # list formatter: use the first field as the list column
    if df is None or df.empty:
        return "I ran the query, but it returned no rows."
    if not fields:
        return "I ran the query, but no fields were provided."

    col = fields[0]
    if col not in df.columns:
        return f"Expected column '{col}' not found in SQL result."

    vals = df[col].dropna().astype(str).tolist()[:max_items]
    items = ", ".join(vals)

    # simplest: template should use {items}
    try:
        return template.format(items=items)
    except Exception:
        return items


def _mask_value(v):
    # masking for SEARCH results (so LLM doesn't see real names)
    s = (v or "").strip()
    if not s:
        return ""
    if len(s) <= 2:
        return "*" * len(s)
    return s[0] + ("*" * (len(s) - 2)) + s[-1]


def _search_candidates(df, pattern, limit_values=5):
    # find text matches in object columns, return masked samples only
    pat = str(pattern)
    out = []

    for c in df.columns:
        s = df[c]
        if s.dtype != "object":
            continue

        ss = s.astype(str)
        mask = ss.str.contains(pat, case=False, na=False)
        vals = ss[mask].dropna().unique().tolist()

        if vals:
            masked = sorted({_mask_value(x) for x in vals})[:limit_values]
            out.append({"column": str(c), "masked_samples": masked, "match_count": len(vals)})

    out.sort(key=lambda x: x["match_count"], reverse=True)
    return {"pattern": pat, "candidates": out[:10]}


def _scalar_from_df(df):
    # if SQL returned a single cell, treat it as scalar answer
    if df.shape == (1, 1):
        v = df.iloc[0, 0]
        if pd.isna(v):
            return None
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                return v
        return v
    return None


def run_agent(df, schema, messages, session_id):
    # main loop: LLM -> action -> tool -> TOOL_RESULT -> repeat
    client = OpenAI()

    tool_result = None
    new_messages = []

    # these are ONLY for user rendering (never sent to LLM as raw data)
    last_rows_for_user = None
    last_scalar_for_user = None

    for step in range(MAX_STEPS):
        prompt = [{"role": "system", "content": SYSTEM}] + list(messages)

        # give the LLM tool results (but only summaries, not raw rows)
        if tool_result is not None:
            prompt.append({"role": "user", "content": "TOOL_RESULT:\n" + json.dumps(tool_result, ensure_ascii=False)})

        try:
            plan = _chat(client, prompt, session_id=session_id, step=step)
        except Exception as e:
            msg = f"LLM error: {e}"
            new_messages.append({"role": "assistant", "content": msg})
            return msg, new_messages, last_rows_for_user

        # validate plan quickly (so we don't crash)
        if not isinstance(plan, dict):
            tool_result = {"error": "plan must be object"}
            continue

        action = plan.get("action")
        if action not in ALLOWED_ACTIONS:
            tool_result = {"error": "invalid action", "plan": plan}
            continue

        # pull common fields
        sql = plan.get("sql")
        pattern = plan.get("pattern")
        msg = (plan.get("message") or "").strip()
        reason = plan.get("reason")

        # ASK: stop and ask user
        if action == "ASK":
            base = msg or "I need one detail to answer that."
            out = f"{base}\n\nReason: {reason}" if reason else base
            new_messages.append({"role": "assistant", "content": out})
            return out, new_messages, last_rows_for_user

        # FINAL: stop and answer user (with optional templating)
        if action == "FINAL":
            template = plan.get("template")
            fill = plan.get("fill") or {}

            display_out = msg
            safe_out_for_history = msg  # store safe only

            if isinstance(template, str) and template:
                src = fill.get("source")

                if src == "sql_first_row":
                    display_out = _format_first_row(template, last_rows_for_user)
                    safe_out_for_history = template

                elif src == "sql_scalar":
                    display_out = _format_scalar(template, last_scalar_for_user)
                    safe_out_for_history = template

                elif src == "sql_rows":
                    fields = fill.get("fields") or []
                    if not isinstance(fields, list):
                        fields = []
                    display_out = _format_rows(template, last_rows_for_user, fields)
                    safe_out_for_history = template

                else:
                    display_out = template
                    safe_out_for_history = template
           
            display_out = _fill_bracket_placeholders(display_out, last_scalar_for_user)

            if last_rows_for_user is not None and not last_rows_for_user.empty:
                display_out = _fill_bracket_placeholders_from_row(display_out, last_rows_for_user.iloc[0].to_dict())

            new_messages.append({"role": "assistant", "content": safe_out_for_history})

            if (not template) and last_rows_for_user is not None and not last_rows_for_user.empty:
                display_out = _fill_bracket_placeholders_from_row(display_out, last_rows_for_user.iloc[0].to_dict())

            return display_out, new_messages, last_rows_for_user

        # SCHEMA: give schema to LLM
        if action == "SCHEMA":
            tool_result = {"schema": schema}
            continue

        # SEARCH: disambiguation
        if action == "SEARCH":
            if not pattern:
                tool_result = {"error": "missing pattern"}
                continue
            tool_result = _search_candidates(df, pattern)
            continue

        # PROFILE: deterministic profiling
        if action == "PROFILE":

            
            kind = (plan.get("profile_kind") or "full").strip().lower()
            col = plan.get("column")
            # keep tiny mapping so SYSTEM words still work
            kind_map = {"summary": "full", "categories": "categorical"}
            kind = kind_map.get(kind, kind)

            # if model asks unsupported profile kinds, just do full (safe fallback)
            SUPPORTED_PROFILE_KINDS = {"full","all","missing","outliers","duplicates","parse","categorical","consistency"}

            kind = (plan.get("profile_kind") or "full").strip().lower()
            kind_map = {"summary": "full", "categories": "categorical"}
            kind = kind_map.get(kind, kind)

            if kind not in SUPPORTED_PROFILE_KINDS:
                kind = "full"


            tool_result = {"profiling": profile_df(df, kind=kind, column=col)}
            last_rows_for_user = None
            last_scalar_for_user = None
            continue

        # SQL: execute query safely
        if action == "SQL":
            if not sql:
                tool_result = {"error": "missing sql"}
                last_rows_for_user = None
                last_scalar_for_user = None
                continue

            err = _validate_sql_rules(sql, for_export=False)
            if err:
                tool_result = {"sql": sql, "error": err}
                last_rows_for_user = None
                last_scalar_for_user = None
                continue

            try:
                outdf = run_sql(df, sql)
                scalar = _scalar_from_df(outdf)

                if scalar is not None:
                    tool_result = {"sql": sql, "scalar": "<computed>"}  # do not leak actual value to LLM
                    last_scalar_for_user = scalar
                    last_rows_for_user = None
                else:
                    tool_result = {"sql": sql, "rows": int(len(outdf)), "columns": list(outdf.columns)}
                    last_rows_for_user = outdf.head(50)
                    last_scalar_for_user = None

            except Exception as e:
                tool_result = {"sql": sql, "error": str(e)}
                last_rows_for_user = None
                last_scalar_for_user = None

            continue

        # EXPORT: run SQL + write xlsx
        if action == "EXPORT":
            if not sql:
                tool_result = {"error": "missing sql for export"}
                last_rows_for_user = None
                last_scalar_for_user = None
                continue

            err = _validate_sql_rules(sql, for_export=True)
            if err:
                tool_result = {"sql": sql, "error": err}
                last_rows_for_user = None
                last_scalar_for_user = None
                continue

            prefix = plan.get("filename_prefix") or "export"

            try:
                outdf = run_sql(df, sql)
                fname = _export_xlsx(outdf, prefix)

                out = f"Created file: {fname}"
                new_messages.append({"role": "assistant", "content": out})
                return out, new_messages, last_rows_for_user

            except Exception as e:
                tool_result = {"sql": sql, "error": f"export failed: {e}"}
                last_rows_for_user = None
                last_scalar_for_user = None
                continue

    # if model didn't finish in time
    out = "Couldn't finish reliably. Please rephrase or specify more details."
    new_messages.append({"role": "assistant", "content": out})
    return out, new_messages, last_rows_for_user
