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

    if 'ORDERDATE' in df.columns:
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')

    query_applied = False
    warnings = []

    groupby_cols = ['COUNTRY']
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

    try:
        plt.figure(figsize=(10, 6))
        plt.bar(df_agg['COUNTRY'], df_agg['Total Sales'])
        plt.xlabel('COUNTRY')
        plt.ylabel('Total Sales')
        plt.title('Sales by Country')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    except Exception as e:
        error = str(e)
        fig = None
        return {
            "title": "Sales by Country",
            "type": "map",
            "why": "Identifies key markets and regional sales variations.",
            "query_applied": query_applied,
            "encodings_used": {"x": {"field": "COUNTRY", "type": "string"}, "y": {"field": "Total Sales", "type": "quantitative"}, "color": {"field": "Total Sales", "type": "quantitative", "legend": True}, "facet": {"field": None, "type": "string"}},
            "columns_used": ['COUNTRY', 'SALES'],
            "warnings": warnings,
            "error": error,
            "data": {"columns": [], "records": [], "row_count": 0}
        }

    fig = plt.gcf()

    data = {
        "columns": df_agg.columns.tolist(),
        "records": df_agg.to_dict(orient="records"),
        "row_count": len(df_agg)
    }

    return {
        "title": "Sales by Country",
        "type": "map",
        "why": "Identifies key markets and regional sales variations.",
        "query_applied": query_applied,
        "encodings_used": {"x": {"field": "COUNTRY", "type": "string"}, "y": {"field": "Total Sales", "type": "quantitative"}, "color": {"field": "Total Sales", "type": "quantitative", "legend": True}, "facet": {"field": None, "type": "string"}},
        "columns_used": ['COUNTRY', 'SALES'],
        "warnings": warnings,
        "error": None,
        "data": data
    }