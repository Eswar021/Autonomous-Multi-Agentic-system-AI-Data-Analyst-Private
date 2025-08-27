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
            df[col] = df[col].astype(str)
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

    if 'ORDERDATE' in df.columns:
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')

    query_applied = False
    warnings = []

    groupby_cols = ["YEAR_ID", "MONTH_ID"]
    df_agg = df.groupby(groupby_cols, dropna=False)["SALES"].sum().reset_index()
    df_agg.rename(columns={"SALES": "Total Sales"}, inplace=True)

    sort_by = ["YEAR_ID", "MONTH_ID"]
    sort_order = [True, True]
    df_agg = df_agg.sort_values(by=sort_by, ascending=sort_order)

    limit = None
    if limit is not None:
        df_agg = df_agg.head(limit)

    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()

    years = df_agg["YEAR_ID"].unique()
    for year in years:
        df_year = df_agg[df_agg["YEAR_ID"] == year]
        ax.plot(df_year["MONTH_ID"], df_year["Total Sales"], label=str(year))

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.set_title("Monthly Sales Trend")
    ax.legend(title="Year")

    fig = plt.gcf()

    columns_used = ['MONTH_ID', 'YEAR_ID', 'SALES']
    encodings_used = ['x', 'y', 'color']

    return {
        "title": "Monthly Sales Trend",
        "type": "line",
        "why": "Helps identify seasonality and trends in sales.",
        "query_applied": query_applied,
        "encodings_used": encodings_used,
        "columns_used": columns_used,
        "warnings": warnings,
        "error": None,
        "data": {
            "columns": df_agg.columns.tolist(),
            "records": df_agg.to_dict(orient="records"),
            "row_count": len(df_agg)
        }
    }