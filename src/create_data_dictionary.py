# File Purpose: Create data dictionary of processed data (\data\processed\subjects_processed.csv)
# Author: Max Freitas
# Last Updated: October 6, 2025

import pandas as pd
import numpy as np
from pathlib import Path
import re

def create_numeric_data_dict(df):
    """
    Create a data dictionary for numeric columns in a dataframe.
    Includes dtype, missing values, unique values, min, max, mean, example.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    data_dict = {}
    
    for col in numeric_cols:
        data_dict[col] = {
            "type": str(df[col].dtype),
            "description": "",  # fill manually
            "num_missing": df[col].isna().sum(),
            "num_unique": df[col].nunique(),
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": df[col].mean(),
            "example": df[col].dropna().iloc[0] if not df[col].isna().all() else None
        }
    
    return pd.DataFrame(data_dict).T

def create_object_data_dict(df):
    """
    Create a data dictionary for object/categorical columns in a dataframe.
    Includes dtype, missing values, unique values, all unique values if <10, example.
    """
    object_cols = df.select_dtypes(include=["object", "category"]).columns
    data_dict = {}
    
    for col in object_cols:
        unique_vals = df[col].dropna().unique()
        # Show all unique values if less than 10, else keep empty
        if len(unique_vals) <= 10:
            unique_display = list(unique_vals)
        else:
            unique_display = []

        data_dict[col] = {
            "type": str(df[col].dtype),
            "description": "",  # fill manually
            "num_missing": df[col].isna().sum(),
            "num_unique": df[col].nunique(),
            "unique_values": unique_display,
            "example": df[col].dropna().iloc[0] if not df[col].isna().all() else None
        }
    
    return pd.DataFrame(data_dict).T




##########################################################
# Testing functions and creating updated .csv file
# Define project root and data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "subjects_processed.csv"



df = pd.read_csv(DATA_PATH)

numeric_dict = create_numeric_data_dict(df)
object_dict = create_object_data_dict(df)

# Save to CSV
numeric_dict.to_csv(PROJECT_ROOT / "data" / "processed" / "tables"/ "numeric_data_dictionary.csv")
object_dict.to_csv(PROJECT_ROOT / "data" / "processed" /  "tables" /"object_data_dictionary.csv")

print("Numeric and object data dictionaries created and saved.")