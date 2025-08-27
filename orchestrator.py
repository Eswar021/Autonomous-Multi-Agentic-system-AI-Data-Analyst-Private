# orchestrator.py
from __future__ import annotations
import io, contextlib, json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from V_SME_DA_MetadataCreator_V1 import generate_and_save_metadata
from V_SME_DA_Planner_V2 import plan_from_metadata
from V_SME_DA_KPI_Executor_V1 import run_kpis_with_llm
from V_SME_DA_Viz_Executor_V1 import run_visualizations_with_llm

def _capture(func, *args, **kwargs) -> Tuple[Any, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = func(*args, **kwargs)
    return out, buf.getvalue()

def run_pipeline(uploaded_file_path: str | Path,
                 objective: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs: Metadata -> Planner -> KPIs -> Viz
    Returns paths + artifacts + logs for UI.
    """
    uploaded_file_path = Path(uploaded_file_path)
    stem = uploaded_file_path.stem

    # Step 1: Metadata
    (metadata, meta_path), meta_logs = _capture(
        generate_and_save_metadata, uploaded_file_path, business_hint=objective or None
    )

    # Step 2: Planner
    (plan, plan_path), plan_logs = _capture(
        plan_from_metadata, dataset_stem=stem, objective=objective or None
    )

    # Step 3: KPIs
    kpi_paths, kpi_logs = _capture(
        run_kpis_with_llm, plan_path
    )
    # Step 4: Visualizations
    viz_paths, viz_logs = _capture(
        run_visualizations_with_llm, plan_path
    )

    # Load KPI and Viz results for UI
    outputs_dir = Path("outputs") / stem
    kpi_results_path = outputs_dir / f"{stem}_kpi_results.json"
    viz_results_path = outputs_dir / f"{stem}_viz_results.json"

    kpis = json.loads(kpi_results_path.read_text(encoding="utf-8")) if kpi_results_path.exists() else []
    vizzes = json.loads(viz_results_path.read_text(encoding="utf-8")) if viz_results_path.exists() else []

    return {
        "stem": stem,
        "meta_path": str(meta_path),
        "plan_path": str(plan_path),
        "kpi_results_path": str(kpi_results_path) if kpi_results_path.exists() else None,
        "viz_results_path": str(viz_results_path) if viz_results_path.exists() else None,
        "metadata": metadata,
        "plan": plan,
        "kpis": kpis,
        "visualizations": vizzes,
        "logs": {
            "metadata": meta_logs,
            "planner": plan_logs,
            "kpis": kpi_logs,
            "visualizations": viz_logs
        }
    }
