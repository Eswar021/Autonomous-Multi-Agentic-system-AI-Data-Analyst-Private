import pandas as pd
import numpy as np
import json
import math
import datetime
import typing
import pathlib

def compute_kpi() -> typing.Dict:
    """
    Computes the number of distinct orders from the sales data.
    """
    file_abspath = "D:\\Autonomous_Dashboard_Creator_Agent_Production\\uploads\\sales_data.csv"
    encoding = "cp1252"
    kpi_name = "Number of Orders"
    kpi_formula = "COUNT(DISTINCT ORDERNUMBER)"
    kpi_why = "Track the volume of sales transactions."
    depends_on_columns = ["ORDERNUMBER"]

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
            "error": f"Failed to read CSV: {str(e)}"
        }

    # Column presence validation
    missing_columns = [col for col in depends_on_columns if col not in df.columns]
    if missing_columns:
        return {
            "name": kpi_name,
            "value": None,
            "formula": kpi_formula,
            "why": kpi_why,
            "details": {},
            "warnings": [],
            "error": f"Missing required columns: {', '.join(missing_columns)}"
        }

    try:
        # Calculate the number of distinct orders
        num_orders = df["ORDERNUMBER"].nunique()

        return {
            "name": kpi_name,
            "value": num_orders,
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