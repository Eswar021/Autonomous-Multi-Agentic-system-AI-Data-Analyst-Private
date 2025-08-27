# ─────────────────────────── Planner / Guider Layer ─────────────────────────────
# Requirements: python-dotenv, langchain-google-genai, pandas
# Uses the same LLM setup from your previous notebook cell

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json, re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

# ─── Config ─────────────────────────────────────────────────────────────────────
METADATA_DIR = Path("metadata")
PLANS_DIR = Path("plans")
PLANS_DIR.mkdir(parents=True, exist_ok=True)

# Reuse your LLM init
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=google_api_key,
    temperature=0.2
)

# 1) Load metadata 
def find_metadata_path(dataset_stem: Optional[str] = None,
                       meta_dir: str | Path = "metadata") -> Path:
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(exist_ok=True, parents=True)
    if dataset_stem:
        p = meta_dir / f"{dataset_stem}_metadata.json"
        if not p.exists():
            raise FileNotFoundError(f"Metadata not found: {p}")
        return p
    cands = sorted(meta_dir.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No metadata files in {meta_dir}")
    return cands[0]

def load_metadata(meta_path: str | Path) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 2) Introspection helpers 
def _col_lookup(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c["name"]: c for c in meta.get("columns", [])}

def _first_name_regex(meta: Dict[str, Any], pat: re.Pattern) -> Optional[str]:
    for c in meta.get("columns", []):
        if pat.search(c["name"]):
            return c["name"]
    return None

def derive_hints(meta: Dict[str, Any]) -> Dict[str, Any]:
    cols = meta.get("columns", [])
    # date
    date_col = None
    for c in cols:
        if "date" in [s.lower() for s in c.get("semantics", [])]:
            date_col = c["name"]; break
    if not date_col:
        date_col = _first_name_regex(meta, re.compile(r"date|dt|timestamp|time", re.I))

    # parts & business-ish
    hints = {
        "date_col": date_col,
        "year_col": _first_name_regex(meta, re.compile(r"\byear\b|YEAR[_]?ID", re.I)),
        "month_col": _first_name_regex(meta, re.compile(r"\bmonth\b|MONTH[_]?ID", re.I)),
        "qtr_col": _first_name_regex(meta, re.compile(r"qtr|quarter", re.I)),
        "sales_like": _first_name_regex(meta, re.compile(r"sales|revenue|amount|price", re.I)),
        "product_like": _first_name_regex(meta, re.compile(r"product(line)?|sku|item|category", re.I)),
        "customer_like": _first_name_regex(meta, re.compile(r"customer(name)?|account|client", re.I)),
        "status_like": _first_name_regex(meta, re.compile(r"status", re.I)),
        "geo_like": (
            _first_name_regex(meta, re.compile(r"country", re.I)) or
            _first_name_regex(meta, re.compile(r"state", re.I)) or
            _first_name_regex(meta, re.compile(r"city", re.I)) or
            _first_name_regex(meta, re.compile(r"territory|region|market", re.I))
        ),
    }
    return hints


# 3) Prompt builder
def build_planner_prompt(meta: Dict[str, Any],
                         hints: Dict[str, Any],
                         objective: Optional[str] = None) -> List[Dict[str, str]]:
    dataset = meta.get("dataset", {})
    cols    = meta.get("columns", [])
    col_min = [
        {k: c[k] for k in ("name", "dtype", "nullable", "semantics")}
        for c in cols
    ]

    sys_prompt = """
      You are an analytics planning agent. Output STRICT JSON ONLY (no markdown, no extra text), matching this schema:

      {
        "objective": "<short business objective or 'none'>",
        "overall_rationale": "<2-5 sentences>",
        "kpis": [
          {"name": "<KPI name>", "formula": "<plain text formula>", "why": "<business value>", "depends_on_columns": ["<col>", "..."]}
        ],
        "analyses": [
          {"name": "<module>", "question": "<business question>", "method": "<brief method>", "expected_insights": "<expected learnings>", "depends_on_columns": ["<col>", "..."]}
        ],
        "visualizations": [
          {
            "title": "<chart title>",
            "purpose": "<what it explains>",
            "why": "<why this chart>",
            "type": "<line|bar|stacked_bar|area|scatter|box|hist|map|heatmap|table>",
            "data_access": { "file_path": "<path>", "file_abspath": "<abspath>" },
            "query": {
              "grain": "<e.g., by YEAR, by MONTH & PRODUCTLINE>",
              "filters": ["<expr>", "..."],
              "aggregations": [{"name":"<metric>", "expr":"<aggregation expr>"}],
              "groupby": ["<col>", "..."],
              "sort": [{"by":"<col or metric>", "order":"asc|desc"}],
              "limit": <int or null>
            },
            "encodings": {
              "x": {"field": "<col or derived>", "type":"temporal|quantitative|string"},
              "y": {"field": "<metric or col>", "type":"quantitative|string"},
              "color": {"field": "<col or null>", "type":"string", "legend": true},
              "facet": {"field": "<col or null>", "type":"string"}
            },
            "columns_used": [{"name":"<col>"}],
            "assumptions": ["<assumption>", "..."],
            "notes_for_codegen": "<hints that will help write code>"
          }
        ],
        "data_quality_checks": [
          {"check":"<name>", "method":"<how to compute>", "tolerance":"<rule>", "columns":["<col>", "..."]}
        ]
      }

      Rules:
      - Do NOT invent columns. Only use the provided columns.
      - If the dataset lacks fields for a specific idea, omit that idea.
      - Tailor the plan strictly to the provided metadata, hints, and objective (if any).
      - Keep names concise; keep "why" actionable.
      - Return valid JSON only.
      """.strip()

    payload = {
        "dataset_name": dataset.get("name"),
        "dataset_paths": {
            "file_path": dataset.get("file_path"),
            "file_abspath": dataset.get("file_abspath"),
            "encoding": dataset.get("file_encoding")
        },
        "row_count": dataset.get("row_count"),
        "column_count": dataset.get("column_count"),
        "dataset_description": meta.get("dataset_description", ""),
        "assumptions": meta.get("assumptions", []),
        "columns_min": col_min,
        "known_hints": hints,
        "objective": objective or "none",
        # keep token usage modest — we do not include raw records here
        "sample_strategy": meta.get("sample", {}).get("strategy", {})
    }

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]


# 4) Validation + enrichment 
def _ensure_list(v):
    return v if isinstance(v, list) else []

def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def enrich_plan_with_metadata(plan: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    No auto-generated analyses/vizzes here — we ONLY enrich what the LLM produced.
    - ensure data_access (paths) present,
    - attach dtype/semantics/description for referenced columns,
    - drop references to non-existent columns.
    """
    ds = meta.get("dataset", {})
    lookup = {c["name"]: c for c in meta.get("columns", [])}

    for viz in plan.get("visualizations", []):
        da = viz.setdefault("data_access", {})
        da.setdefault("file_path", ds.get("file_path"))
        da.setdefault("file_abspath", ds.get("file_abspath"))

        # collect referenced columns (best-effort from fields)
        referenced = set()
        # from query
        q = viz.get("query", {})
        for g in _ensure_list(q.get("groupby")):
            if g in lookup: referenced.add(g)
        for f in _ensure_list(q.get("filters")):
            m = re.match(r"\s*([A-Za-z0-9_]+)\s*[<>=~!]", str(f))
            if m and m.group(1) in lookup:
                referenced.add(m.group(1))
        for agg in _ensure_list(q.get("aggregations")):
            expr = (agg.get("expr") or "")
            for col in lookup.keys():
                # naive containment; safe enough for enrichment
                if re.search(rf"\b{re.escape(col)}\b", expr):
                    referenced.add(col)

        # from encodings
        enc = viz.get("encodings", {})
        for key in ("x", "y", "color", "facet"):
            fld = _safe_get(enc, key, "field")
            if isinstance(fld, str) and fld in lookup:
                referenced.add(fld)

        # plus explicitly listed
        for cu in viz.get("columns_used", []):
            nm = cu.get("name")
            if isinstance(nm, str) and nm in lookup:
                referenced.add(nm)

        # rebuild columns_used with rich info
        viz["columns_used"] = [{
            "name": nm,
            "dtype": lookup[nm].get("dtype"),
            "semantics": lookup[nm].get("semantics", []),
            "description": lookup[nm].get("description", "")
        } for nm in sorted(referenced)]

        viz.setdefault("notes_for_codegen", "Handle date parsing and nulls as appropriate.")

    return plan

def parse_and_validate_llm_json(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(json)?", "", text, flags=re.I).strip()
    plan = json.loads(cleaned)  # let it raise if invalid
    # minimal schema checks
    for key in ("kpis", "analyses", "visualizations"):
        if key not in plan or not isinstance(plan[key], list):
            raise ValueError(f"Planner JSON missing list '{key}'")
    return plan


# 5) Runner 
def plan_from_metadata(dataset_stem: Optional[str] = None,
                       meta_dir: str | Path = "metadata",
                       objective: Optional[str] = None,
                       out_dir: str | Path = "plans") -> Tuple[Dict[str, Any], Path]:
    """
    Load metadata, ask LLM to generate the plan (no heuristic fallback),
    validate JSON, enrich with metadata details, and save.
    """
    meta_path = find_metadata_path(dataset_stem, meta_dir=meta_dir)
    meta = load_metadata(meta_path)
    hints = derive_hints(meta)

    messages = build_planner_prompt(meta, hints, objective)
    resp = llm.invoke(messages)
    txt = resp.content if hasattr(resp, "content") else str(resp)

    plan = parse_and_validate_llm_json(txt)
    plan = enrich_plan_with_metadata(plan, meta)

    # stamp provenance and save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = meta.get("dataset", {}).get("name", "dataset")
    out_path = out_dir / f"{stem}_plan.json"

    plan["_meta"] = {
        "metadata_path": str(meta_path),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
        "objective": objective or "none",
        "model": "gemini-2.0-flash"
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    return plan, out_path

if __name__ == "__main__":
    # demo / manual test only
    pass

    # # 6) Example 
    # plan, pth = plan_from_metadata(dataset_stem="sales_data",
    #                             objective="Increase sales and identify growth opportunities in 2005.")
    # print("Saved plan:", pth)
    # print(json.dumps(plan["visualizations"][:2], indent=2)[:1200] + " ...")