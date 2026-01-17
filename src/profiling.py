# Simple, generic profiling for one pandas DataFrame.
# Safe-by-default: returns only aggregates (no raw rows), masks categorical samples.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# Treat these as "missing" in text columns
_MISSING_STRINGS = {"", " ", "nan", "null", "none", "na", "n/a", "nil", "-"}

def _mask_value(v: str) -> str:
    #Mask a string value (shows shape without leaking exact text).
    s = (v or "").strip()
    if not s:
        return ""
    if len(s) <= 2:
        return "*" * len(s)
    return s[0] + ("*" * (len(s) - 2)) + s[-1]


def _is_missing_series(s: pd.Series) -> pd.Series:
    #NaN is missing for all dtypes, for text also treat common null tokens as missing.
    miss = s.isna()
    if s.dtype == "object":
        ss = s.astype(str).str.strip().str.lower()
        miss = miss | ss.isin(_MISSING_STRINGS)
    return miss


def _try_parse_datetime(non_missing: pd.Series) -> Tuple[int, int]:
    #Return (valid, invalid) datetime parses on non-missing values.
    parsed = pd.to_datetime(non_missing, errors="coerce", utc=False)
    valid = int(parsed.notna().sum())
    invalid = int(parsed.isna().sum())
    return valid, invalid


def _try_parse_numeric(non_missing: pd.Series) -> Tuple[int, int]:
    #Samae for numeric
    coerced = pd.to_numeric(non_missing, errors="coerce")
    valid = int(coerced.notna().sum())
    invalid = int(coerced.isna().sum())
    return valid, invalid

def _profile_missing(df: pd.DataFrame, column: Optional[str] = None) -> Dict[str, Any]:
    #Missing counts + percent per column.
    rows = int(len(df))

    def one_col(c: str) -> Dict[str, Any]:
        miss = _is_missing_series(df[c])
        m = int(miss.sum())
        return {"name": c, "missing_total": m, "missing_pct": (m / rows * 100.0) if rows else 0.0}

    if column:
        return {"rows": rows, "missing": [one_col(column)]}

    cols = [one_col(str(c)) for c in df.columns]
    cols.sort(key=lambda x: x["missing_total"], reverse=True)
    return {"rows": rows, "missing": cols}


def _iqr_outliers(s: pd.Series) -> Dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {"outliers": 0, "lower_bound": None, "upper_bound": None, "min": None, "max": None}

    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = int(((x < lower) | (x > upper)).sum())
    return {
        "outliers": outliers,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _profile_outliers(df: pd.DataFrame, column: Optional[str] = None) -> Dict[str, Any]:
    # Outliers for numeric columns (or one chosen column).
    if column:
        targets = [column]
    else:
        targets = []
        for c in df.columns:
            if pd.api.types.is_bool_dtype(df[c]):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                targets.append(str(c))

    out = [{"name": c, **_iqr_outliers(df[c])} for c in targets]
    out.sort(key=lambda x: x.get("outliers", 0), reverse=True)
    return {"outliers": out}



def _duplicates_summary(df: pd.DataFrame) -> Dict[str, Any]:

    # Generic duplicates
    # duplicate rows (exact full-row duplicates)

    rows = int(len(df))
    if rows == 0:
        return {"rows": 0, "dup_rows": 0, "dup_rows_pct": 0.0, "dup_by": []}

    dup_rows = int(df.duplicated().sum())

    dup_by: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        miss = _is_missing_series(s)
        s2 = s[~miss]

        if s2.empty:
            continue

        # Light normalization for text
        if s2.dtype == "object":
            s2 = s2.astype(str).str.strip()

        vc = s2.value_counts()
        duplicate_keys = int((vc > 1).sum())

        if duplicate_keys > 0:
            dup_by.append({"key": str(col), "duplicate_keys": duplicate_keys})

    dup_by.sort(key=lambda x: x["duplicate_keys"], reverse=True)

    return {
        "rows": rows,
        "dup_rows": dup_rows,
        "dup_rows_pct": (dup_rows / rows * 100.0),
        "dup_by": dup_by,
    }


def _parse_quality(df: pd.DataFrame) -> Dict[str, Any]:

    # Generic parsing sanity:
    # Try datetime + numeric parsing on each column

    rows = int(len(df))
    cols: List[Dict[str, Any]] = []

    for c in df.columns:
        name = str(c)
        s = df[c]
        miss = _is_missing_series(s)
        non_missing = s[~miss]
        n = int(len(non_missing))
        if n == 0:
            continue

        item: Dict[str, Any] = {"name": name}

        # datetime parse (generic: try it, report only when mixed)
        dt_valid, dt_invalid = _try_parse_datetime(non_missing)
        if dt_valid > 0 and dt_invalid > 0:
            item["date_parse_invalid"] = dt_invalid
            item["date_parse_invalid_pct"] = dt_invalid / n * 100.0

        # numeric parse (generic: try it, report only when mixed)
        num_valid, num_invalid = _try_parse_numeric(non_missing)
        if num_valid > 0 and num_invalid > 0:
            item["numeric_parse_invalid"] = num_invalid
            item["numeric_parse_invalid_pct"] = num_invalid / n * 100.0

        if "date_parse_invalid" in item or "numeric_parse_invalid" in item:
            cols.append(item)

    cols.sort(
        key=lambda x: int(x.get("date_parse_invalid", 0)) + int(x.get("numeric_parse_invalid", 0)),
        reverse=True,
    )
    return {"rows": rows, "columns": cols}

def _categorical_summary(df: pd.DataFrame, max_cols: int = 12, top_k: int = 5) -> Dict[str, Any]:
    #Categorical columns: unique count + top masked values.
    out: List[Dict[str, Any]] = []
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]

    for c in obj_cols[:max_cols]:
        s = df[c].dropna().astype(str).str.strip()
        if s.empty:
            continue

        uniq = int(s.nunique(dropna=True))
        vc = s.value_counts().head(top_k)

        top_vals = [{"value": _mask_value(str(v)), "count": int(cnt)} for v, cnt in vc.items()]
        out.append({"name": str(c), "unique": uniq, "top_values": top_vals})

    return {"categorical": out}


def _consistency_checks(df: pd.DataFrame) -> Dict[str, Any]:

    # Generic consistency checks:
    # - negative values in numeric columns
    # - constant columns (only 1 unique non-missing value)

    rows = int(len(df))
    if rows == 0:
        return {"rows": 0, "checks": []}

    checks: List[Dict[str, Any]] = []

    # Negative numeric values
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        miss = x.isna()
        bad = (~miss) & (x < 0)
        bad_n = int(bad.sum())
        base_n = int((~miss).sum())
        if base_n > 0 and bad_n > 0:
            checks.append({
                "name": "negative_values",
                "column": str(c),
                "bad_rows": bad_n,
                "checked_rows": base_n,
                "bad_pct": (bad_n / base_n * 100.0),
            })

    # Constant columns
    for c in df.columns:
        s = df[c]
        miss = _is_missing_series(s)
        s2 = s[~miss]
        if s2.empty:
            continue
        uniq = int(s2.nunique(dropna=True))
        if uniq == 1:
            checks.append({"name": "constant_column", "column": str(c)})

    return {"rows": rows, "checks": checks}


def _profile_full(df: pd.DataFrame) -> Dict[str, Any]:
    #Full profiling

    rows = int(len(df))
    cols_n = int(df.shape[1])

    missing = _profile_missing(df)["missing"]

    types = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]

    outliers = _profile_outliers(df)["outliers"]
    dups = _duplicates_summary(df)
    parse = _parse_quality(df)
    cat = _categorical_summary(df)
    consistency = _consistency_checks(df)

    return {
        "rows": rows,
        "columns_count": cols_n,
        "columns": missing,         # used by llm.py to show "worst columns"
        "types": types,
        "outliers": outliers,
        "duplicates": dups,
        "parse_quality": parse,
        "categorical": cat,
        "consistency": consistency,
    }

# Public entrypoint (KEEP THIS API)

def profile_df(
    df: pd.DataFrame,
    kind: Optional[str] = None,
    column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Entry point used by llm.py.

    kind:
      - full/all
      - missing
      - outliers
      - duplicates
      - parse
      - categorical
      - consistency
    """
    k = (kind or "full").strip().lower()
    col = (column or "").strip() or None

    if col is not None and col not in df.columns:
        return {"error": f"unknown column: {col}", "available_columns": [str(c) for c in df.columns]}

    if k in ("full", "all"):
        return _profile_full(df)

    if k == "missing":
        return _profile_missing(df, column=col)

    if k == "outliers":
        return _profile_outliers(df, column=col)

    if k == "duplicates":
        return {"duplicates": _duplicates_summary(df)}

    if k == "parse":
        return {"parse_quality": _parse_quality(df)}

    if k == "categorical":
        return _categorical_summary(df)

    if k == "consistency":
        return _consistency_checks(df)

    return {"error": f"unknown profile kind: {k}"}
