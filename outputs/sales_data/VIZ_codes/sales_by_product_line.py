import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import datetime
import typing
import pathlib

def build_viz():
    file_abspath = "D:\\Autonomous_Dashboard_Creator_Agent_Production\\uploads\\sales_data.csv"
    encoding = "cp1252"
    df = pd.read_csv(file_abspath, encoding=encoding)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "ORDERDATE" in df.columns:
        df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")

    query_applied = ""
    warnings = []

    groupby_cols = ["PRODUCTLINE"]
    df_agg = df.groupby(groupby_cols, dropna=False)["SALES"].sum().reset_index()
    df_agg.rename(columns={"SALES": "Total Sales"}, inplace=True)

    sort_by = "Total Sales"
    sort_order = "desc"
    if sort_order == "asc":
        ascending = True
    else:
        ascending = False
    df_agg = df_agg.sort_values(by=sort_by, ascending=ascending)

    limit = None
    if limit is not None:
        df_agg = df_agg.head(limit)

    x_field = "PRODUCTLINE"
    y_field = "Total Sales"

    fig, ax = plt.subplots()
    ax.bar(df_agg[x_field], df_agg[y_field])
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title("Sales by Product Line")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig = plt.gcf()

    data = {
        "columns": df_agg.columns.tolist(),
        "records": df_agg.to_dict(orient="records"),
        "row_count": len(df_agg)
    }

    encodings_used = {
        "x": {"field": "PRODUCTLINE", "type": "string"},
        "y": {"field": "Total Sales", "type": "quantitative"},
        "color": {"field": None, "type": "string", "legend": True},
        "facet": {"field": None, "type": "string"}
    }

    columns_used = df.columns.tolist()

    result = {
        "title": "Sales by Product Line",
        "type": "bar",
        "why": "Identifies top-selling product categories.",
        "query_applied": query_applied,
        "encodings_used": encodings_used,
        "columns_used": columns_used,
        "warnings": warnings,
        "error": None,
        "data": data
    }

    return result