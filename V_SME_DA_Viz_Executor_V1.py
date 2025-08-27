# ====================== Visualization Agent (minimal patch) ======================
from __future__ import annotations
import json, re, io, ast, textwrap, contextlib, datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LLM (Gemini via langchain-google-genai)
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------- LLM init ----------------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=google_api_key,
    temperature=0.2
)

# ---------------------------- IO helpers ----------------------------
def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(obj: Any, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def slugify(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', "_", (s or "")).strip("_").lower()[:80]

# ---------------------------- Prompt utils (unchanged) ----------------------------
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

def _compact_records(records: List[Dict[str, Any]], max_rows=24, max_chars=120) -> List[Dict[str, Any]]:
    rows = records[:max_rows]
    out = []
    for r in rows:
        o = {}
        for k,v in r.items():
            s = "" if v is None else str(v)
            o[k] = s[:max_chars]
        out.append(o)
    return out

def build_viz_prompt(viz: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    ds = metadata["dataset"]
    file_abspath = ds.get("file_abspath") or ds.get("file_path") or ds.get("file")
    file_encoding = ds.get("file_encoding") or "utf-8"
    dataset_name = ds.get("name", "dataset")

    cols_desc = _columns_for_prompt(metadata)
    sample = metadata.get("sample", {})
    sample_records = _compact_records(sample.get("records", []), max_rows=20)

    title = viz.get("title", "")
    viz_type = viz.get("type", "")
    why = viz.get("why", "")
    query = viz.get("query", {}) or {}
    encodings = viz.get("encodings", {}) or {}

    sys_rules = """
You are a code generator. Output ONLY valid Python code (no prose/markdown).
Write one zero-argument function named `build_viz()` that:
- Imports only: pandas as pd, numpy as np, matplotlib.pyplot as plt, json, math, datetime, typing, pathlib.
- Reads the CSV from the provided absolute path using the given encoding.
- Coerces numeric fields with pd.to_numeric(..., errors="coerce").
- If ORDERDATE exists, parse with pd.to_datetime(..., errors="coerce").
- Apply filters. Operators order: "==", "!=", ">=", "<=", ">", "<", "=".
- For SUM(SALES) use:
    df_agg = df.groupby(groupby_cols, dropna=False)["SALES"].sum().reset_index()
    df_agg.rename(columns={"SALES": "Total Sales"}, inplace=True)
- Sort by requested keys; respect limit if provided.
- Create a Matplotlib figure and plot with ax.plot(...) or ax.bar(...)/ax.barh(...).
- Call `fig = plt.gcf()` at end (do NOT save/show).
- Return a JSON-serializable dict with keys:
  title, type, why, query_applied, encodings_used, columns_used, warnings, error, data {columns, records, row_count}
Prohibitions: no file writes, no network, no env access, no subprocess.
    """.strip()

    user_payload = {
        "dataset": {
            "name": dataset_name,
            "file_abspath": file_abspath,
            "encoding": file_encoding
        },
        "schema": cols_desc,
        "visualization": {
            "title": title,
            "type": viz_type,
            "why": why,
            "query": query,
            "encodings": encodings
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
Implement `build_viz()` using only the given schema columns.

file_abspath = {json.dumps(file_abspath)}
encoding = {json.dumps(file_encoding)}

Visualization:
- title = {json.dumps(title)}
- type  = {json.dumps(viz_type)}
- why   = {json.dumps(why)}

Query JSON:
{json.dumps(query, ensure_ascii=False, indent=2)}

Encodings JSON:
{json.dumps(encodings, ensure_ascii=False, indent=2)}

End with `fig = plt.gcf()`; do not save/show.
""".strip()

    prompt = (
        "SYSTEM RULES:\n" + sys_rules + "\n\n"
        "INPUT JSON (metadata + visualization):\n" + json.dumps(user_payload, ensure_ascii=False, indent=2) + "\n\n"
        + dev_instructions
    )
    return prompt

# ---------------------------- LLM call (unchanged) ----------------------------
def ask_llm_for_viz_code(llm, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You write Python code only."},
        {"role": "user", "content": prompt},
    ]
    resp = llm.invoke(messages)
    text = getattr(resp, "content", str(resp)).strip()
    text = re.sub(r"^```(?:python)?\s*|\s*```$", "", text, flags=re.I|re.M)
    return text.strip()

# ---------------------------- Validation (CHANGED: more permissive) ----------------------------
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

_ALLOWED_IMPORTS = {"pandas", "numpy", "json", "math", "datetime", "typing", "pathlib", "matplotlib"}

def validate_viz_code(code: str) -> Tuple[bool, str]:
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
        if isinstance(node, ast.FunctionDef) and node.name == "build_viz":
            if func_found:
                return False, "Multiple build_viz functions defined."
            if (len(node.args.args) != 0) or node.args.vararg or node.args.kwarg:
                return False, "build_viz must take zero arguments."
            func_found = True

    if not func_found:
        return False, "build_viz() not found."

    # NOTE: removed the strict "must contain plotting call" requirement.
    # The orchestrator will fallback-plot if needed.

    return True, ""

# ---------------------------- Sandbox exec (CHANGED: add set) ----------------------------
def run_viz_in_sandbox(code: str) -> tuple[dict | None, str | None, str]:
    """
    Execute generated code with restricted builtins. Returns (result_dict, error, captured_stdout)
    """
    safe_globals = {
        "__builtins__": {
            # allow normal imports inside generated code
            "__import__": __import__,       # <-- critical fix

            # safe builtins
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
            "set": set,
            "print": print,
        },
        # prebound libs (so code can use them even if it forgot to import)
        "pd": pd,
        "np": np,
        "plt": plt,
        "json": json,
        "math": __import__("math"),
        "datetime": datetime,
        "typing": __import__("typing"),
        "pathlib": __import__("pathlib"),
    }
    safe_locals: dict[str, Any] = {}

    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, safe_globals, safe_locals)
            fn = safe_locals.get("build_viz") or safe_globals.get("build_viz")
            if not callable(fn):
                return None, "build_viz not callable after exec", stdout_buf.getvalue()
            result = fn()
        if not isinstance(result, dict):
            return None, "build_viz did not return a dict", stdout_buf.getvalue()
        json.dumps(result)  # serializable check
        return result, None, stdout_buf.getvalue()
    except Exception as e:
        return None, f"Execution error: {e}", stdout_buf.getvalue()

# ---------------------------- Orchestrator (CHANGED: fallback plot + save) ----------------------------
def run_visualizations_with_llm(
    plan_path: str | Path,
    metadata_path: Optional[str | Path] = None,
    out_root: str | Path = "outputs",
    verbose: bool = True,
    save_code_snippets: bool = True,
) -> Dict[str, Path]:
    plan = load_json(plan_path)
    meta_path = metadata_path or plan.get("_meta", {}).get("metadata_path")
    if not meta_path:
        raise ValueError("metadata_path not provided and not present in plan._meta.metadata_path.")
    metadata = load_json(meta_path)

    dataset_name = metadata["dataset"]["name"]
    ds_dir = Path(out_root) / dataset_name
    code_dir = ds_dir / "VIZ_codes"
    viz_dir = ds_dir / "visualizations"
    fig_dir = viz_dir / "figures"
    data_dir = viz_dir / "data"
    if save_code_snippets:
        code_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    fn_registry: List[Dict[str, Any]] = []

    for i, viz in enumerate(plan.get("visualizations", []), start=1):
        title = viz.get("title", f"viz_{i}")
        vslug = slugify(title)
        print("="*100)
        print(f"[VIZ {i}] {title}  |  type={viz.get('type','')}")
        print("-"*100)

        prompt = build_viz_prompt(viz, metadata)
        code = ask_llm_for_viz_code(llm, prompt)

        # Show code in notebook (debug)
        print("Generated code:\n")
        print(code)
        print("-"*100)

        valid, why_bad = validate_viz_code(code)
        if not valid:
            print(f"Validation: ❌ {why_bad}\n")
            result_entry = {
                "title": title,
                "type": viz.get("type"),
                "why": viz.get("why"),
                "error": f"Validation failed: {why_bad}",
            }
            results.append(result_entry)
            if save_code_snippets:
                (code_dir / f"{vslug}.py").write_text(code, encoding="utf-8")
            continue

        # Save code
        if save_code_snippets:
            (code_dir / f"{vslug}.py").write_text(code, encoding="utf-8")

        # Track figures before exec
        figs_before = set(plt.get_fignums())

        # Execute
        result, exec_err, stdout = run_viz_in_sandbox(code)
        if stdout.strip():
            print("[stdout]\n" + stdout)

        fig_path = None
        data_path = None

        if exec_err:
            print(f"Execution: ❌ {exec_err}")
            results.append({
                "title": title,
                "type": viz.get("type"),
                "why": viz.get("why"),
                "error": exec_err,
                "data_path": None,
                "figure_path": None,
                "_stdout": stdout[:3000]
            })
            continue

        print("Execution: ✅ success")

        # Save filtered/aggregated data (if any)
        dat = result.get("data") or {}
        dcols = dat.get("columns") or []
        drows = dat.get("records") or []
        try:
            df_out = pd.DataFrame(drows, columns=dcols) if dcols else pd.DataFrame(drows)
        except Exception:
            df_out = pd.DataFrame(drows)

        data_path = data_dir / f"{vslug}.csv"
        if df_out is not None:
            df_out.to_csv(data_path, index=False, encoding="utf-8")
        else:
            pd.DataFrame().to_csv(data_path, index=False, encoding="utf-8")

        # Detect new figure; if none, fallback-plot
        figs_after = set(plt.get_fignums())
        new_figs = sorted(list(figs_after - figs_before))

        if not new_figs and (df_out is not None) and (not df_out.empty):
            enc = viz.get("encodings", {}) or {}
            x_field = (enc.get("x") or {}).get("field")
            y_field = (enc.get("y") or {}).get("field")
            try:
                if x_field in df_out.columns and y_field in df_out.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if viz.get("type") in ("bar", "map"):
                        ax.bar(df_out[x_field], df_out[y_field])
                    else:
                        ax.plot(df_out[x_field], df_out[y_field])
                    ax.set_xlabel(x_field)
                    ax.set_ylabel(y_field)
                    ax.set_title(title)
                    plt.tight_layout()
                    new_figs = [plt.get_fignums()[-1]]
            except Exception as e:
                print(f"[Warn] Fallback plotting failed: {e}")

        if new_figs:
            fig = plt.figure(new_figs[-1])
            fig_path = fig_dir / f"{vslug}.png"
            fig.savefig(fig_path, dpi=120, bbox_inches="tight")
            print(f"[Figure saved] {fig_path}")
            plt.show()
            plt.close(fig)
        else:
            print("[Warn] No matplotlib figure detected.")

        # Record result
        res = {
            "title": result.get("title", title),
            "type": result.get("type", viz.get("type")),
            "why": result.get("why", viz.get("why")),
            "query_applied": result.get("query_applied"),
            "encodings_used": result.get("encodings_used"),
            "columns_used": result.get("columns_used"),
            "warnings": result.get("warnings"),
            "error": result.get("error"),
            "data_path": str(data_path) if data_path else None,
            "figure_path": str(fig_path) if fig_path else None,
            "_stdout": stdout[:3000] if stdout else None,
        }
        results.append(res)

        # Function registry
        fn_registry.append({
            "viz_slug": vslug,
            "title": title,
            "type": viz.get("type"),
            "code_path": str((code_dir / f"{vslug}.py").resolve()),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        })

    # Save final artifacts
    results_path = ds_dir / f"{dataset_name}_viz_results.json"
    save_json(results, results_path)

    fn_path = ds_dir / f"{dataset_name}_viz_functions.json"
    save_json(fn_registry, fn_path)

    print("="*100)
    print(f"[LLM VIZ] Saved results: {results_path}")
    print(f"[LLM VIZ] Saved function registry: {fn_path}")
    print("="*100)
    return {"viz_results_path": results_path, "viz_functions_path": fn_path}


if __name__ == "__main__":
    # demo / manual test only
    pass

    # # ---------------------------- Example usage ----------------------------
    # plan_file = "plans/sales_data_plan.json"
    # paths = run_visualizations_with_llm(plan_file)
    # print(paths)

    # print("\nVIZ_Codes:")
    # for p in sorted((Path("outputs")/load_json("metadata/sales_data_metadata.json")["dataset"]["name"]/"VIZ_codes").glob("*.py")):
    #     print(p)