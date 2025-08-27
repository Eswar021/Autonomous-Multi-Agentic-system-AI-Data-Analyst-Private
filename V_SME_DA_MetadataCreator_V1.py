# 0) Imports & LLM setup

import os, io, re, json, math, textwrap
from pathlib import Path
from datetime import datetime,timezone
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np

# LLM (Gemini via langchain-google-genai)
# pip install python-dotenv langchain-google-genai pandas pyarrow openpyxl
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

# import warnings
# warnings.filterwarnings("ignore", message="Could not infer format")
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=google_api_key,
    temperature=0.3
)
# print("=== testing LLM connection ===")
# LLM_RESPONSE = llm.invoke("hi")
# print(LLM_RESPONSE.content)

# 1) File loading (CSV / XLSX / Parquet)
def read_any_table(path: str | Path, max_preview_rows: int = 200_000) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1"]

    if suffix in [".csv", ".tsv", ".txt"]:
        last_err = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(path, nrows=max_preview_rows, encoding=enc)
                df._source_encoding = enc  # attach for later
                return df
            except Exception as e:
                last_err = e
                continue
        raise last_err
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, nrows=max_preview_rows)
        df._source_encoding = "excel"
        return df
    elif suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
        df._source_encoding = "parquet"
        return df
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

# 2) Pandas→string dtype mapping + semantic heuristics
_PANDAS_DTYPE_MAP = {
    "int64": "integer",
    "Int64": "integer_nullable",
    "float64": "float",
    "bool": "boolean",
    "object": "string",
    "string": "string",
    "datetime64[ns]": "datetime",
    "category": "category",
}

def map_dtype(series: pd.Series) -> str:
    dt = str(series.dtype)
    if "datetime64" in dt:
        return "datetime"
    return _PANDAS_DTYPE_MAP.get(dt, dt)

_money_re = re.compile(r"(^|\b)(amount|amt|price|cost|revenue|sales|net|gross|subtotal|total|usd|inr|eur)\b", re.I)
_pct_re   = re.compile(r"(^|\b)(pct|percent|percentage|ratio|rate)\b", re.I)
_date_re  = re.compile(r"(date|dt|timestamp|time|at$)", re.I)
_id_re    = re.compile(r"(id|uuid|guid|key|number|no$)", re.I)
_geo_re   = re.compile(r"(country|state|city|region|geo|market|territory)", re.I)
_qty_re   = re.compile(r"(qty|quantity|units|count|volume)", re.I)
_prod_re  = re.compile(r"(product|sku|item|material|part|name)", re.I)
_curr_symbol_re = re.compile(r"^[\$\₹\£\€]")

def detect_semantics(col: str, s: pd.Series) -> List[str]:
    sem = []
    lname = col.lower()

    # name-based
    if _date_re.search(lname): sem.append("date")
    if _id_re.search(lname): sem.append("id")
    if _money_re.search(lname): sem.append("money")
    if _pct_re.search(lname): sem.append("pct")
    if _geo_re.search(lname): sem.append("geo")
    if _qty_re.search(lname): sem.append("quantity")
    if _prod_re.search(lname): sem.append("product")

    # value-based hints
    sample_non_null = s.dropna().head(50).astype(str)
    if not sem and any(_curr_symbol_re.match(v.strip()) for v in sample_non_null):
        sem.append("money")
    if "string" == map_dtype(s) and s.nunique(dropna=True) == len(s.dropna()):
        sem.append("unique_like")

    # date coercion attempt
    # date coercion attempt (quiet + tolerant)
    if "date" not in sem:
        dt_try = pd.to_datetime(s.dropna().head(50), errors="coerce")
        if dt_try.notna().mean() >= 0.7:
            sem.append("date_like")

    return sorted(list(set(sem)))

# 3) Data profiling summary (nulls, uniques, ranges) 

def profile_dataframe(df: pd.DataFrame, sample_for_range: int = 50_000) -> Dict[str, Any]:
    n_rows = len(df)
    col_profiles = []

    for col in df.columns:
        s = df[col]
        dtype = map_dtype(s)
        non_null = s.notna().sum()
        null_pct = float((n_rows - non_null) / max(n_rows, 1) * 100.0)
        unique_ct = s.nunique(dropna=True)
        unique_pct = float(unique_ct / max(non_null, 1) * 100.0)
        semantics = detect_semantics(col, s)

        stats: Dict[str, Any] = {
            "name": col,
            "dtype": dtype,
            "nullable": null_pct > 0.0,
            "null_pct": round(null_pct, 4),
            "unique_count": int(unique_ct),
            "unique_pct": round(unique_pct, 4),
            "semantics": semantics,
        }

        # numeric ranges
        if dtype in ("integer", "integer_nullable", "float"):
            nums = pd.to_numeric(s, errors="coerce").dropna()
            if len(nums) > 0:
                smpl = nums.sample(min(len(nums), sample_for_range), random_state=42)
                stats["min"] = float(np.nanmin(smpl))
                stats["max"] = float(np.nanmax(smpl))
                stats["mean"] = float(np.nanmean(smpl))
                stats["p50"] = float(np.nanpercentile(smpl, 50))
        col_profiles.append(stats)

    return {
        "row_count": int(n_rows),
        "column_count": int(df.shape[1]),
        "columns": col_profiles,
    }



# 4) Safe sample extraction for LLM (truncate long strings & rows)

def sample_for_llm(df: pd.DataFrame, max_rows: int = 15, max_chars: int = 120) -> Dict[str, List[str]]:
    """
    Small, simple sample for the prompt payload. Keeps token usage low.
    """
    smpl = df.head(max_rows).copy()
    for c in smpl.columns:
        smpl[c] = smpl[c].astype(str).str.slice(0, max_chars)
    rows = smpl.to_dict(orient="records")
    return {"columns": list(smpl.columns), "records": rows}


### Smart, diverse sampling for downstream agents (rich sample) 
def _primary_date_col(profile: Dict[str, Any]) -> str | None:
    """
    Choose the primary date column using semantics, then a name fallback.
    """
    for c in profile.get("columns", []):
        if "date" in [s.lower() for s in c.get("semantics", [])]:
            return c["name"]
    for c in profile.get("columns", []):
        if "date" in c["name"].lower():
            return c["name"]
    return None


def _good_category_col(df: pd.DataFrame, profile: Dict[str, Any]) -> str | None:
    """
    Prefer 'product' or 'geo' semantics with moderate cardinality (5..100).
    Fallback to any string/category in that range.
    """
    candidates = []
    for c in profile.get("columns", []):
        name = c["name"]
        card = c.get("unique_count", 0)
        sems = [s.lower() for s in c.get("semantics", [])]
        if name not in df.columns:
            continue
        if c.get("dtype") in ("string", "category") and 5 <= card <= 100:
            weight = (("product" in sems) * 3) + (("geo" in sems) * 2) + 1
            candidates.append((weight, -card, name))
    if not candidates:
        for c in profile.get("columns", []):
            name = c["name"]
            card = c.get("unique_count", 0)
            if name in df.columns and c.get("dtype") in ("string", "category") and 5 <= card <= 100:
                candidates.append((1, -card, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _truncate_for_llm(df: pd.DataFrame, max_chars: int = 120) -> pd.DataFrame:
    """
    Truncate long strings to keep payloads compact.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == "object":
            out[c] = out[c].astype(str).str.slice(0, max_chars)
    return out


def _to_python_scalar(x):
    """
    Make numpy/pandas scalars JSON-safe. NaN/NaT -> None.
    """
    # pandas NA-like → None
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def build_rich_sample(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    random_rows: int = 20,
    time_bins: int = 6,
    time_rows_per_bin: int = 3,
    cat_top_k: int = 5,
    cat_rows_per_value: int = 3,
    max_chars: int = 120,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Build a composite, diverse sample to help downstream analysis:
      - head(5)
      - random N rows
      - time-sliced rows across date quantiles (if a date column exists)
      - category-sliced rows for a representative categorical

    Returns:
      {
        "strategy": {...},
        "columns": [...],
        "records": [ {JSON-safe rows...} ]
      }
    """
    parts = []

    # 1) Head sample
    parts.append(df.head(5))

    # 2) Random sample
    if len(df) > 0:
        parts.append(df.sample(min(random_rows, len(df)), random_state=random_state))

    # 3) Time-sliced sample (using primary date col)
    date_col = _primary_date_col(profile)
    if date_col and date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        tmp = df.copy()
        tmp["_sample_dt"] = dt
        tmp = tmp[tmp["_sample_dt"].notna()]
        if len(tmp) > 0:
            # Use qcut over datetime ordinals; guard for low-unique cases
            try:
                ordinals = tmp["_sample_dt"].astype("int64")
                tmp["_time_bin"] = pd.qcut(
                    ordinals,
                    q=min(time_bins, max(1, tmp["_sample_dt"].nunique())),
                    duplicates="drop"
                )
                for _, g in tmp.groupby("_time_bin", observed=True):
                    parts.append(
                        g.drop(columns=["_time_bin"], errors="ignore")
                         .sample(min(time_rows_per_bin, len(g)), random_state=random_state)
                    )
            except Exception:
                # Fallback: monthly grouping
                tmp["_month"] = tmp["_sample_dt"].dt.to_period("M")
                for _, g in tmp.groupby("_month", observed=True):
                    parts.append(g.sample(min(time_rows_per_bin, len(g)), random_state=random_state))
        # cleanup (best-effort)
        for col in ["_sample_dt", "_time_bin", "_month"]:
            if col in tmp.columns:
                tmp.drop(columns=[col], inplace=True, errors="ignore")

    # 4) Category-sliced sample
    cat_col = _good_category_col(df, profile)
    if cat_col:
        vc = df[cat_col].astype(str).value_counts().head(cat_top_k)
        for val in vc.index:
            g = df[df[cat_col].astype(str) == val]
            parts.append(g.sample(min(cat_rows_per_value, len(g)), random_state=random_state))

    # Combine, drop duplicates, and truncate long strings
    combo = (pd.concat(parts, axis=0, ignore_index=True).drop_duplicates()
             if parts else df.head(10))
    combo = _truncate_for_llm(combo, max_chars=max_chars)

    # ---- JSON-safe conversion (fixes NaT/NaN/np types) ----
    safe = combo.copy()
    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            dt = pd.to_datetime(safe[c], errors="coerce", utc=True)
            # ISO 8601 with Z; keep None for NaT
            safe[c] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(dt.notna(), None)
        else:
            # None for NA, then map numpy scalars to Python
            safe[c] = safe[c].where(pd.notna(safe[c]), None)
            safe[c] = safe[c].map(_to_python_scalar)

    rows = safe.head(200).to_dict(orient="records")  # hard cap

    strategy = {
        "head": 5,
        "random": min(random_rows, len(df)),
        "date_column": date_col,
        "time_bins": time_bins if date_col else 0,
        "time_rows_per_bin": time_rows_per_bin if date_col else 0,
        "category_column": cat_col,
        "cat_top_k": cat_top_k if cat_col else 0,
        "cat_rows_per_value": cat_rows_per_value if cat_col else 0,
        "max_chars_per_cell": max_chars,
        "random_state": random_state
    }

    return {"strategy": strategy, "columns": list(safe.columns), "records": rows}

# 5) LLM prompts to describe dataset & columns 
# ─── 5) LLM prompts to describe dataset & columns (STRICT JSON + robust fallback) ──
def _normalize_key(s: str) -> str:
    # Normalize for fuzzy matching (UPPER + strip non-alnum/_)
    return re.sub(r"[^A-Za-z0-9_]+", "", s or "").upper()

def _map_found_to_actual(found: dict, actual_cols: list[str]) -> dict:
    """Map descriptions from found keys to actual columns by normalized matching."""
    actual_norm = {_normalize_key(c): c for c in actual_cols}
    out = {}
    for k, v in found.items():
        nk = _normalize_key(k)
        if nk in actual_norm and isinstance(v, str) and v.strip():
            out[actual_norm[nk]] = v.strip()
    return out

def _parse_markdown_column_bullets(text: str, actual_cols: list[str]) -> dict:
    """
    Extract patterns like:
      * **ORDERNUMBER:** description
      - ORDERNUMBER: description
      • PRODUCTLINE: description
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pairs = {}

    # Pattern 1: **COLNAME:** desc
    pat_bold = re.compile(r"^\*+\s*\*\*([^*]+?)\*\*\s*:\s*(.+)$")
    # Pattern 2: -/• COLNAME: desc
    pat_plain = re.compile(r"^[-•]\s*([A-Za-z0-9_ ]+?)\s*:\s*(.+)$")

    for ln in lines:
        m = pat_bold.match(ln)
        if m:
            col, desc = m.group(1).strip(), m.group(2).strip()
            pairs[col] = desc
            continue
        m = pat_plain.match(ln)
        if m:
            col, desc = m.group(1).strip(), m.group(2).strip()
            pairs[col] = desc

    # Also handle inline bullets separated by " * " within a paragraph
    if not pairs:
        # Split on " * " then parse segments like "**COL:** desc"
        segs = re.split(r"\s\*\s", text)
        for seg in segs:
            m = pat_bold.search(seg)
            if m:
                col, desc = m.group(1).strip(), m.group(2).strip()
                pairs[col] = desc

    return _map_found_to_actual(pairs, actual_cols)

def llm_describe_dataset_and_columns(
    filename: str,
    profile: Dict[str, Any],
    sample: Dict[str, Any],
    business_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns ONLY model-provided descriptions. If a column wasn't described by the LLM,
    we leave it blank (no defaults).
    """
    sys_prompt = """
        You are a senior data analyst. You MUST return STRICT JSON ONLY (no prose, no markdown), with keys:
        {
        "dataset_description": "3-6 clear sentences describing the dataset in business terms.",
        "column_descriptions": { "<EXACT_COLUMN_NAME>": "one concise line per column", ... },
        "assumptions": ["explicit assumptions or uncertainties", ...]
        }
        Rules:
        - The keys in "column_descriptions" MUST MATCH the provided column names EXACTLY.
        - Do not invent columns. Do not rename columns.
        - If unsure about a column, write a cautious, short description (no more than 140 chars).
        - No additional keys, no code fences, no markdown.
        """.strip()

    user_payload = {
        "filename": filename,
        "row_count": profile.get("row_count"),
        "columns_profile": profile.get("columns"),
        "sample": sample,
        "column_names": [c["name"] for c in profile.get("columns", [])],
        "business_hint": business_hint or "General sales analytics"
    }

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]

    resp = llm.invoke(messages)
    text = resp.content if hasattr(resp, "content") else str(resp)

    # 1) Try strict JSON first
    try:
        cleaned = re.sub(r"```(json)?", "", text, flags=re.I).strip()
        data = json.loads(cleaned)
        ds_desc = str(data.get("dataset_description", "")).strip()
        col_desc = data.get("column_descriptions", {}) or {}
        assumptions = data.get("assumptions", []) or []

        # Keep ONLY LLM-provided descriptions (no defaults).
        # Also ensure we don't include keys for columns the LLM didn't mention.
        actual_cols = [c["name"] for c in profile.get("columns", [])]
        col_desc = _map_found_to_actual(col_desc, actual_cols)
        return {
            "dataset_description": ds_desc,
            "column_descriptions": col_desc,
            "assumptions": assumptions
        }
    except Exception:
        pass

    # 2) Fallback: parse markdown bullets (your current output format)
    actual_cols = [c["name"] for c in profile.get("columns", [])]
    col_desc = _parse_markdown_column_bullets(text, actual_cols)

    # Try to extract a dataset description heuristically from the first paragraph
    # (no defaults—only if we can pull something meaningful)
    ds_desc = ""
    # grab first sentence-ish chunk
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paras:
        ds_desc = paras[0]
        # trim obvious headings tokens
        ds_desc = re.sub(r"^\s*okay,?\s*here.*?:\s*", "", ds_desc, flags=re.I)
        ds_desc = re.sub(r"\*\*?\d\.\s*Dataset Description\*\*?:?\s*", "", ds_desc, flags=re.I)

    # Assumptions: pull bullet lines under a section named 'Assumptions'
    assumptions = []
    for block in paras:
        if re.search(r"assumption", block, flags=re.I):
            for ln in block.splitlines():
                if re.match(r"^\s*[-–•\*]\s+", ln):
                    assumptions.append(re.sub(r"^\s*[-–•\*]\s+", "", ln).strip())
            break

    return {
        "dataset_description": ds_desc[:2000],
        "column_descriptions": col_desc,  # ONLY what we extracted from the LLM
        "assumptions": assumptions[:10],
    }

# 6) Build the combined DSM-like metadata 
def build_metadata_json(
    file_path: str | Path,
    df: pd.DataFrame,
    profile: Dict[str, Any],
    llm_notes: Dict[str, Any],
    project: str = "agentic_sales_analyst",
    sample_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    p = Path(file_path)
    cols = []
    for cprof in profile["columns"]:
        name = cprof["name"]
        cols.append({
            "name": name,
            "dtype": cprof["dtype"],
            "nullable": cprof["nullable"],
            "null_pct": cprof["null_pct"],
            "unique_count": cprof["unique_count"],
            "unique_pct": cprof["unique_pct"],
            "semantics": cprof["semantics"],
            "description": llm_notes.get("column_descriptions", {}).get(name, "")
        })

    return {
        "dataset": {
            "name": p.stem,
            "source": "user_upload",
            "project": project,
            "row_count": profile["row_count"],
            "column_count": profile["column_count"],
            "file": p.name,
            "file_path": str(p),
            "file_abspath": str(p.resolve()),
            "file_encoding": getattr(df, "_source_encoding", None),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        },
        "columns": cols,
        "dataset_description": llm_notes.get("dataset_description", ""),
        "assumptions": llm_notes.get("assumptions", []),
        "sample": sample_payload or {},   # <— include diverse sample here
        "quality_summary": {
            "warnings": [
                "Descriptions are inferred from small samples; verify critical fields.",
                "Currency and units may be unspecified unless clearly present."
            ]
        }
    }


# 7) Save to metadata/<stem>_metadata.json
# ─── Save to metadata/<stem>_metadata.json and embed the path ───────────────────
def save_metadata(metadata: Dict[str, Any], out_dir: str | Path = "metadata") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = metadata["dataset"]["name"]
    out_path = out_dir / f"{stem}_metadata.json"

    # write paths into the metadata object BEFORE persisting
    metadata["dataset"]["metadata_path"] = str(out_path)
    metadata["dataset"]["metadata_abspath"] = str(out_path.resolve())

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return out_path


# 8) One-call convenience wrapper 
def generate_and_save_metadata(
    file_path: str | Path,
    business_hint: Optional[str] = None,
    max_rows_for_llm: int = 15
) -> Tuple[Dict[str, Any], Path]:
    df = read_any_table(file_path)
    profile = profile_dataframe(df)
    # small sample for LLM explanation step (kept small)
    llm_sample = sample_for_llm(df, max_rows=max_rows_for_llm)
    llm_notes = llm_describe_dataset_and_columns(str(file_path), profile, llm_sample, business_hint=business_hint)

    # rich, diverse sample for downstream agents
    rich_sample = build_rich_sample(df, profile)

    metadata = build_metadata_json(file_path, df, profile, llm_notes, sample_payload=rich_sample)
    out_path = save_metadata(metadata)
    return metadata, out_path


if __name__ == "__main__":
    # demo / manual test only
    pass

    # # 9) Example usage (run this cell)
    # metadata, path = generate_and_save_metadata(
    #     r"data\sales_data.csv",
    #     business_hint="Sample Sales Data, Order Info, Sales, Customer, Shipping, etc., Used for Segmentation, Customer Analytics, Clustering and More. Inspired for retail analytics."
    # )
    # print(f"Saved: {path}")
    # print(json.dumps(metadata["dataset"], indent=2))



