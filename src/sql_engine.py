# sql_engine.py
# Runs SELECT queries on a DataFrame exposed as table `data`.

import re
import pandas as pd
from pandasql import sqldf

_BLOCKLIST = re.compile(r"\b(drop|delete|update|insert|alter|create|attach|pragma|vacuum)\b", re.I)

def run_sql(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("single statement only")
    if not s.lower().startswith("select"):
        raise ValueError("SELECT only")
    if _BLOCKLIST.search(s):
        raise ValueError("unsafe keyword")
    return sqldf(s, {"data": df})
