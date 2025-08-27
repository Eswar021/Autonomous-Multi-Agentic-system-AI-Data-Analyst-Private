# KPI Agent (LLM-generated code)
from __future__ import annotations
import json, re, textwrap, ast, io, contextlib, builtins, datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

# --- LLM init (same as you had) ---
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=google_api_key,
    temperature=0.2
)

# ---------- Helpers ----------
def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(obj: Any, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def slugify(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', "_", s).strip("_").lower()[:80]

# ---------- Prompt Builder ----------
def _columns_for_prompt(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    cols = []
    for c in metadata.get("columns", []):
        cols.append({
            "name": c["name"],
            "dtype": c.get("dtype"),
            "nullable": c.get("nullable"),
            "semantics": c.get("semantics", []),
            "desc": c.get("description", "")
        })
    return cols

def _compact_records(records: List[Dict[str, Any]], max_rows=30, max_chars=120) -> List[Dict[str, Any]]:
    rows = records[:max_rows]
    out = []
    for r in rows:
        o = {}
        for k,v in r.items():
            s = "" if v is None else str(v)
            o[k] = s[:max_chars]
        out.append(o)
    return out

def build_kpi_prompt(kpi: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    ds = metadata["dataset"]
    file_abspath = ds.get("file_abspath") or ds.get("file_path") or ds.get("file")
    file_encoding = ds.get("file_encoding") or "utf-8"
    dataset_name = ds.get("name", "dataset")
    rows = metadata.get("row_count") or metadata.get("dataset", {}).get("row_count")
    cols_desc = _columns_for_prompt(metadata)
    sample = metadata.get("sample", {})
    sample_records = _compact_records(sample.get("records", []), max_rows=28, max_chars=120)

    objective = (kpi.get("why") or "").strip()
    kname = kpi.get("name", "").strip()
    formula = kpi.get("formula", "").strip()
    depends = kpi.get("depends_on_columns", [])

    # Strict code-only instruction with constraints and return contract
    sys_rules = """
You are a code generator. Output ONLY valid Python code, no prose, no markdown.
Constraints:
- Define exactly one zero-argument function named `compute_kpi()`.
- Do NOT read any inputs or env vars; embed constants you need.
- Allowed imports only: pandas as pd, numpy as np, json, math, datetime, typing, pathlib.
- Forbidden: os, sys, subprocess, shutil, requests, http, sockets, file writes, deleting files, shell.
- No network, no external I/O besides reading the provided CSV path.
- Return a JSON-serializable dict with keys:
  {
    "name": <str>,
    "value": <number or None>,
    "formula": <str>,
    "why": <str>,
    "details": { ... optional breakdowns ... },
    "warnings": [ ... ],
    "error": <str or None>
  }
Robustness:
- Load CSV from `file_abspath` with `encoding`.
- Handle missing columns gracefully, set "error" with explanation.
- Convert numeric columns safely (coerce errors).
- If `formula` references year filters like YEAR_ID = 2005, apply them safely.
- If `formula` is an arithmetic expression of earlier KPIs, compute from the data directly if possible; otherwise return error.
- Avoid changing global state.
    """.strip()

    user_payload = {
        "dataset": {
            "name": dataset_name,
            "file_abspath": file_abspath,
            "encoding": file_encoding,
            "row_count": rows,
        },
        "schema": cols_desc,
        "kpi": {
            "name": kname,
            "formula": formula,
            "why": objective,
            "depends_on_columns": depends
        },
        "hints": {
            "date_columns": [c["name"] for c in metadata.get("columns", []) if "date" in [s.lower() for s in c.get("semantics", [])]],
            "money_columns": [c["name"] for c in metadata.get("columns", []) if "money" in [s.lower() for s in c.get("semantics", [])]],
        },
        "sample_preview": {
            "columns": sample.get("columns", []),
            "records": sample_records
        }
    }

    dev_instructions = f"""
Task: Generate only Python code for `compute_kpi()`, a zero-arg function that:
- Reads the dataset at: file_abspath = {json.dumps(file_abspath)}
- Uses encoding: {json.dumps(file_encoding)}
- Computes KPI: {json.dumps(kname)}
- Formula: {json.dumps(formula)}
- Why: {json.dumps(objective)}
- Uses only columns present in the schema; validate presence.
- If filtering by year/month is implied in formula (e.g., YEAR_ID = 2005), apply it.
- Coerce numeric cols with `pd.to_numeric(..., errors="coerce")`.
- If ORDERDATE is needed, parse with `pd.to_datetime(..., errors="coerce")`.
- Return the dict exactly as specified (no printing).
"""
    prompt = (
        "SYSTEM RULES:\n" + sys_rules + "\n\n"
        "INPUT JSON (metadata + KPI):\n" + json.dumps(user_payload, ensure_ascii=False, indent=2) + "\n\n"
        + dev_instructions
    )
    return prompt

# ---------- LLM Call ----------
def ask_llm_for_kpi_code(llm, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You write Python code only."},
        {"role": "user", "content": prompt},
    ]
    resp = llm.invoke(messages)
    text = getattr(resp, "content", str(resp)).strip()
    # Strip accidental code fences if present
    text = re.sub(r"^```(?:python)?\s*|\s*```$", "", text, flags=re.I|re.M)
    return text.strip()

# ---------- Code Validation ----------
_FORBIDDEN_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b",
    r"\bopen\s*\(",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\b__import__\s*\(",
    r"\bsocket\b",
    r"\brequests\b",
    r"\bhttp\.client\b",
]
_ALLOWED_IMPORTS = {"pandas", "numpy", "json", "math", "datetime", "typing", "pathlib"}

def validate_generated_code(code: str) -> Tuple[bool, str]:
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            return False, f"Forbidden construct matched: {pat}"
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"Code is not valid Python: {e}"

    func_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    return False, f"Disallowed import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in _ALLOWED_IMPORTS:
                return False, f"Disallowed import: from {node.module} import ..."
        if isinstance(node, ast.FunctionDef) and node.name == "compute_kpi":
            if func_found:
                return False, "Multiple compute_kpi functions defined."
            # must have zero args (no *args/**kwargs either)
            if (len(node.args.args) != 0) or node.args.vararg or node.args.kwarg:
                return False, "compute_kpi must take zero arguments."
            func_found = True
    if not func_found:
        return False, "compute_kpi() not found."
    return True, ""

# ---------- Sandbox Execution (with safe __import__) ----------
def _make_safe_import(allowed_roots: set[str]):
    real_import = builtins.__import__
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root not in allowed_roots:
            raise ImportError(f"Import of '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)
    return _safe_import

def run_in_sandbox(code: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """
    Execute generated code with restricted builtins. Returns (result_dict, error, captured_stdout)
    """
    safe_import = _make_safe_import(_ALLOWED_IMPORTS)

    safe_globals = {
        "__builtins__": {
            # safe subset
            "__import__": safe_import,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "range": range,
            "abs": abs,
            "round": round,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "dict": dict,
            "list": list,
            "print": print,  # captured
        },
        # expose whitelisted libs
        "pd": pd,
        "np": np,
        "json": json,
    }
    safe_locals: Dict[str, Any] = {}
    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, safe_globals, safe_locals)
            fn = safe_locals.get("compute_kpi") or safe_globals.get("compute_kpi")
            if not callable(fn):
                return None, "compute_kpi not callable after exec", stdout_buf.getvalue()
            result = fn()
        if not isinstance(result, dict):
            return None, "compute_kpi did not return a dict", stdout_buf.getvalue()
        json.dumps(result)  # ensure serializable
        return result, None, stdout_buf.getvalue()
    except Exception as e:
        return None, f"Execution error: {e}", stdout_buf.getvalue()

# ---------- Main: Run all KPIs with LLM-generated code ----------
def run_kpis_with_llm(
    plan_path: str | Path,
    metadata_path: Optional[str | Path] = None,
    out_root: str | Path = "outputs",
    save_code_snippets: bool = True,
    llm_temperature: float = 0.2,
    verbose: bool = True,
    stdout_max: int = 2000,
) -> Dict[str, Path]:
    plan = load_json(plan_path)
    meta_path = metadata_path or plan.get("_meta", {}).get("metadata_path")
    if not meta_path:
        raise ValueError("metadata_path not provided and not present in plan._meta.metadata_path.")
    metadata = load_json(meta_path)

    dataset_name = metadata["dataset"]["name"]
    ds_dir = Path(out_root) / dataset_name
    code_dir = ds_dir / "KPI_codes"
    if save_code_snippets:
        code_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    kpi_function_registry: List[Dict[str, Any]] = []

    # Optionally nudge temperature
    try:
        if hasattr(llm, "temperature"):
            llm.temperature = llm_temperature
    except Exception:
        pass

    model_name = getattr(llm, "model", "gemini-2.0-flash")  # best-effort

    for idx, kpi in enumerate(plan.get("kpis", []), start=1):
        kslug = slugify(kpi.get("name", f"kpi_{idx}"))
        prompt = build_kpi_prompt(kpi, metadata)
        code = ask_llm_for_kpi_code(llm, prompt)

        if verbose:
            print("\n" + "="*90)
            print(f"[KPI {idx}] {kpi.get('name')}")
            print("-"*90)
            print("Generated code:\n")
            print(code[:10_000])  # show whole or cap to be safe
            print("-"*90)

        valid, why_bad = validate_generated_code(code)
        if not valid:
            if save_code_snippets:
                (code_dir / f"{kslug}.py").write_text(code, encoding="utf-8")
            res = {
                "name": kpi.get("name"),
                "formula": kpi.get("formula"),
                "why": kpi.get("why"),
                "error": f"Validation failed: {why_bad}",
                "_diagnostic": {"snippet_head": code[:500]}
            }
            results.append(res)
            if verbose:
                print(f"Validation: ❌ {why_bad}")
            # Register function metadata (failed but stored)
            kpi_function_registry.append({
                "name": kpi.get("name"),
                "slug": kslug,
                "code_path": str((code_dir / f"{kslug}.py").resolve()) if save_code_snippets else None,
                "description": kpi.get("why") or "",
                "formula": kpi.get("formula") or "",
                "created_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "model": model_name,
                "plan_path": str(Path(plan_path).resolve()),
                "metadata_path": str(Path(meta_path).resolve()),
                "status": "validation_failed"
            })
            continue

        result, exec_err, stdout = run_in_sandbox(code)
        if save_code_snippets:
            (code_dir / f"{kslug}.py").write_text(code, encoding="utf-8")

        if exec_err:
            entry = {
                "name": kpi.get("name"),
                "formula": kpi.get("formula"),
                "why": kpi.get("why"),
                "error": exec_err,
                "stdout": (stdout or "")[:stdout_max]
            }
            results.append(entry)
            status = "execution_error"
        else:
            # Attach minimal provenance
            result["_provenance"] = {
                "kpi_slug": kslug,
                "metadata_path": str(meta_path),
                "plan_path": str(plan_path)
            }
            if stdout and stdout.strip():
                result["_stdout"] = stdout[:stdout_max]
            results.append(result)
            status = "ok"

        if verbose:
            if exec_err:
                print(f"Execution: ❌ {exec_err}")
            else:
                print("Execution: ✅ success")
            if stdout:
                print("\n[Captured stdout]")
                print(stdout[:stdout_max])
            print("-"*90)
            print("[Result JSON]")
            print(json.dumps(results[-1], indent=2)[:10_000])

        # Register function metadata
        kpi_function_registry.append({
            "name": kpi.get("name"),
            "slug": kslug,
            "code_path": str((code_dir / f"{kslug}.py").resolve()) if save_code_snippets else None,
            "description": kpi.get("why") or "",
            "formula": kpi.get("formula") or "",
            "created_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": model_name,
            "plan_path": str(Path(plan_path).resolve()),
            "metadata_path": str(Path(meta_path).resolve()),
            "status": status
        })

    # Save results and KPI function registry
    results_path = ds_dir / f"{dataset_name}_kpi_results.json"
    save_json(results, results_path)

    kpi_meta_path = ds_dir / f"{dataset_name}_kpi_functions.json"
    save_json(kpi_function_registry, kpi_meta_path)

    if verbose:
        print("\n" + "="*90)
        print(f"[LLM KPI] Saved results: {results_path}")
        print(f"[LLM KPI] Saved function registry: {kpi_meta_path}")
        print("="*90)

    return {"results_path": results_path, "kpi_functions_path": kpi_meta_path}


### =========================================================================== ###
if __name__ == "__main__":
    # demo / manual test only
    pass

    # # ---------------- Example usage ----------------
    # plan_file = "plans/sales_data_plan.json"
    # outputs = run_kpis_with_llm(plan_file, verbose=True)
    # print(outputs)