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
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if "ORDERDATE" in df.columns:
        df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors='coerce')

    query_applied = False
    warnings = []

    groupby_cols = ["DEALSIZE"]
    df_agg = df.groupby(groupby_cols, dropna=False)["SALES"].sum().reset_index()
    df_agg.rename(columns={"SALES": "Total Sales"}, inplace=True)

    sort_by = "Total Sales"
    sort_order = "desc"
    if sort_by:
        ascending = sort_order != "desc"
        df_agg = df_agg.sort_values(by=sort_by, ascending=ascending)

    limit = None
    if limit:
        df_agg = df_agg.head(limit)

    plt.figure(figsize=(10, 6))
    plt.bar(df_agg["DEALSIZE"], df_agg["Total Sales"])
    plt.xlabel("DEALSIZE")
    plt.ylabel("Total Sales")
    plt.title("Sales by Deal Size")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig = plt.gcf()

    columns_used = ["DEALSIZE", "SALES"]
    encodings_used = ["x", "y"]

    data = {
        "columns": df_agg.columns.tolist(),
        "records": df_agg.to_dict(orient="records"),
        "row_count": len(df_agg)
    }

    result = {
        "title": "Sales by Deal Size",
        "type": "bar",
        "why": "Understand the impact of deal size on overall sales performance.",
        "query_applied": query_applied,
        "encodings_used": encodings_used,
        "columns_used": columns_used,
        "warnings": warnings,
        "error": None,
        "data": data
    }

    return result