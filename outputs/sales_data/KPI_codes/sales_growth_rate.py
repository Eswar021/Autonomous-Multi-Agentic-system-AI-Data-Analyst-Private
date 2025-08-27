import pandas as pd
import numpy as np
import json
import math
import datetime
import typing
import pathlib

def compute_kpi():
    file_abspath = "D:\\Autonomous_Dashboard_Creator_Agent_Production\\uploads\\sales_data.csv"
    encoding = "cp1252"
    kpi_name = "Sales Growth Rate"
    kpi_formula = "(Current Period Sales - Previous Period Sales) / Previous Period Sales"
    kpi_why = "Shows the percentage change in sales over time."
    depends_on_columns = ["SALES", "ORDERDATE"]
    
    try:
        df = pd.read_csv(file_abspath, encoding=encoding)
    except Exception as e:
        return {
            "name": kpi_name,
            "value": None,
            "formula": kpi_formula,
            "why": kpi_why,
            "details": {},
            "warnings": [],
            "error": f"Error reading CSV: {str(e)}"
        }

    # Validate column presence
    missing_columns = [col for col in depends_on_columns if col not in df.columns]
    if missing_columns:
        return {
            "name": kpi_name,
            "value": None,
            "formula": kpi_formula,
            "why": kpi_why,
            "details": {},
            "warnings": [],
            "error": f"Missing columns: {', '.join(missing_columns)}"
        }

    # Data type conversions
    df["SALES"] = pd.to_numeric(df["SALES"], errors="coerce")
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")

    # Aggregate sales by date (year-month)
    df["YearMonth"] = df["ORDERDATE"].dt.to_period("M")
    sales_by_month = df.groupby("YearMonth")["SALES"].sum().sort_index()

    if len(sales_by_month) < 2:
        return {
            "name": kpi_name,
            "value": None,
            "formula": kpi_formula,
            "why": kpi_why,
            "details": {},
            "warnings": ["Insufficient data to calculate growth rate (less than 2 periods)."],
            "error": None
        }

    # Calculate sales growth rate
    current_period_sales = sales_by_month.iloc[-1]
    previous_period_sales = sales_by_month.iloc[-2]

    if previous_period_sales == 0:
        growth_rate = None
        warning = ["Previous period sales are zero, cannot calculate growth rate."]
    else:
        growth_rate = (current_period_sales - previous_period_sales) / previous_period_sales
        warning = []

    return {
        "name": kpi_name,
        "value": growth_rate if growth_rate is not None else None,
        "formula": kpi_formula,
        "why": kpi_why,
        "details": {
            "current_period_sales": current_period_sales,
            "previous_period_sales": previous_period_sales
        },
        "warnings": warning,
        "error": None
    }