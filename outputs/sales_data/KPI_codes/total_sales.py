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
    kpi_name = "Total Sales"
    kpi_formula = "SUM(SALES)"
    kpi_why = "Measures overall revenue generated."
    depends_on_columns = ["SALES"]
    
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

    try:
        df["SALES"] = pd.to_numeric(df["SALES"], errors="coerce")
        total_sales = df["SALES"].sum()
        
        return {
            "name": kpi_name,
            "value": total_sales,
            "formula": kpi_formula,
            "why": kpi_why,
            "details": {},
            "warnings": [],
            "error": None
        }

    except Exception as e:
        return {
            "name": kpi_name,
            "value": None,
            "formula": kpi_formula,
            "why": kpi_why,
            "details": {},
            "warnings": [],
            "error": f"Error computing KPI: {str(e)}"
        }