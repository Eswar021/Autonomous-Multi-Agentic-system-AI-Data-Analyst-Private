# app.py
import os
from pathlib import Path
import json
import streamlit as st
import pandas as pd

from orchestrator import run_pipeline

st.set_page_config(page_title="Autonomous Data Analyst", layout="wide")

st.sidebar.header("Upload & Configure")
uploaded = st.sidebar.file_uploader("Upload a CSV/XLSX/Parquet", type=["csv","xlsx","xls","parquet"])
objective = st.sidebar.text_input("Business objective (optional)", value="")
run_btn = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

st.title("🤖 Autonomous Data Analyst")
st.caption("Upload a dataset → click **Run analysis** → get KPIs, goals, visualizations. Expand debug if needed.")

progress = st.empty()
tabs = st.tabs(["Overview", "KPIs", "Visualizations", "Debug"])

def _save_upload(f) -> Path:
    up_dir = Path("uploads"); up_dir.mkdir(parents=True, exist_ok=True)
    p = up_dir / f.name
    with open(p, "wb") as out:
        out.write(f.getbuffer())
    return p

if run_btn:
    if not uploaded:
        st.error("Please upload a file first.")
        st.stop()

    # Save uploaded file
    progress.progress(5, text="Saving file…")
    path = _save_upload(uploaded)

    # Run pipeline
    try:
        progress.progress(15, text="Generating metadata…")
        result = run_pipeline(path, objective=(objective or None))
    except Exception as e:
        st.exception(e)
        st.stop()

    progress.progress(100, text="Done ✔")

    # ---------- Overview ----------
    with tabs[0]:
        meta = result["metadata"]
        plan = result["plan"]

        left, right = st.columns([2,1], gap="large")
        with left:
            st.subheader("Dataset")
            ds = meta.get("dataset", {})
            st.write(
                f"**Name:** {ds.get('name')}  \n"
                f"**Rows:** {ds.get('row_count'):,}  \n"
                f"**Columns:** {ds.get('column_count')}  \n"
                f"**Source:** {ds.get('source')}  \n"
                f"**Encoding:** {ds.get('file_encoding')}"
            )
            if meta.get("dataset_description"):
                st.markdown("**Description**")
                st.write(meta["dataset_description"])
            if meta.get("assumptions"):
                st.markdown("**Assumptions**")
                st.write("\n".join(f"• {a}" for a in meta["assumptions"]))

        with right:
            st.subheader("Goals")
            obj = plan.get("objective") or (result.get("plan", {}).get("_meta", {}).get("objective"))
            st.write(obj or "—")
            if plan.get("overall_rationale"):
                st.markdown("**Rationale**")
                st.write(plan["overall_rationale"])

        if meta.get("quality_summary", {}).get("warnings"):
            st.info("**Data quality notes**\n\n" + "\n".join(
                f"- {w}" for w in meta["quality_summary"]["warnings"]
            ))

    # ---------- KPIs ----------
    with tabs[1]:
        kpis = result["kpis"]
        if not kpis:
            st.write("No KPIs were produced.")
        else:
            # Summarize core fields
            rows = []
            for k in kpis:
                rows.append({
                    "KPI": k.get("name"),
                    "Value": k.get("value"),
                    "Formula": k.get("formula"),
                    "Why": k.get("why"),
                    "Error": k.get("error")
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            errs = [k for k in kpis if k.get("error")]
            if errs:
                with st.expander("KPI errors (expand)"):
                    st.json(errs)

    # ---------- Visualizations ----------
    with tabs[2]:
        vizzes = result["visualizations"]
        if not vizzes:
            st.write("No visualizations produced.")
        else:
            for v in vizzes:
                with st.container(border=True):
                    st.markdown(f"### {v.get('title') or 'Visualization'}")
                    if v.get("why"):
                        st.caption(v["why"])
                    fig_path = v.get("figure_path")
                    if fig_path and Path(fig_path).exists():
                        st.image(str(fig_path), use_column_width=True)
                    else:
                        st.warning("No figure available.")
                    # Offer data behind the viz
                    data_path = v.get("data_path")
                    if data_path and Path(data_path).exists():
                        with open(data_path, "rb") as f:
                            st.download_button("Download data (CSV)", data=f, file_name=Path(data_path).name)
                    if v.get("error"):
                        st.error(v["error"])

    # ---------- Debug ----------
    with tabs[3]:
        st.markdown("Use these only if you want to inspect the pipeline.")
        with st.expander("Metadata (logs)", expanded=False):
            st.code(result["logs"]["metadata"] or "")
            st.markdown(f"**Saved metadata:** `{result['meta_path']}`")
        with st.expander("Planner (logs)", expanded=False):
            st.code(result["logs"]["planner"] or "")
            st.markdown(f"**Saved plan:** `{result['plan_path']}`")
        with st.expander("KPIs (logs)", expanded=False):
            st.code(result["logs"]["kpis"] or "")
            st.markdown(f"**KPI results:** `{result.get('kpi_results_path')}`")
        with st.expander("Visualizations (logs)", expanded=False):
            st.code(result["logs"]["visualizations"] or "")
            st.markdown(f"**Viz results:** `{result.get('viz_results_path')}`")
