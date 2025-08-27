import pandas as pd
import numpy as np
import json
import math
import datetime
import typing
import pathlib

def compute_kpi() -> typing.Dict[str, typing.Any]:
    """
    Computes the Average Order Value KPI from the sales_data dataset.
    """
    file_abspath = "D:\\Autonomous_Dashboard_Creator_Agent_Production\\uploads\\sales_data.csv"
    encoding = "cp1252"
    kpi_name = "Average Order Value"
    kpi_formula = "SUM(SALES) / COUNT(ORDERNUMBER)"
    kpi_why = "Indicates the average revenue per order."
    depends_on_columns = ["SALES", "ORDERNUMBER"]

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
            "error": f"Error reading the CSV file: {str(e)}",
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
            "error": f"Missing columns in the dataset: {', '.join(missing_columns)}",
        }

    # Convert to numeric, coercing errors
    df["SALES"] = pd.to_numeric(df["SALES"], errors="coerce")
    df["ORDERNUMBER"] = pd.to_numeric(df["ORDERNUMBER"], errors="coerce")

    # Drop rows with NaN in relevant columns after numeric conversion
    df = df.dropna(subset=["SALES", "ORDERNUMBER"])

    try:
        total_sales = df["SALES"].sum()
        order_count = df["ORDERNUMBER"].nunique()  # Count unique order numbers

        if order_count == 0:
            average_order_value = 0  # Or None, depending on desired behavior
        else:
            average_order_value = total_sales / order_count

        return {
            "name": kpi_name,
            "value": average_order_value,
            "formula": kpi_formula,
            "why": kpi_why,
            "details": {
                "total_sales": total_sales,
                "order_count": order_count,
            },
            "warnings": [],
            "error": None,
        }

    except Exception as e:
        return {
            "name": kpi_name,
            "value": None,
            "formula": kpi_formula,
            "why": kpi_why,
            "details": {},
            "warnings": [],
            "error": f"Error calculating the KPI: {str(e)}",
        }