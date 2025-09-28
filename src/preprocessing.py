# File Purpose: Cleaning raw data (\data\raw\subjects.csv)
# Author: Max Freitas
# Last Updated: September 28, 2025

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer

# Define root(s)
PROJECT_ROOT = Path(__file__).resolve().parents[1]   
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "subjects.csv"

def load_data(path=DATA_PATH):
    """Load raw subject data into a pandas DataFrame."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found at {path}")
        return None


# check subjects
df=load_data(path=DATA_PATH)

# fix consistency of string/object cols by making all upper case
def uppercase_all_object_columns(df, print_unique=False):
    """
    Converts all object/string columns in a DataFrame to uppercase.
    
    Parameters:
        df (pd.DataFrame): input dataframe
        print_unique (bool): if True, print unique values after conversion
    Returns:
        pd.DataFrame: dataframe with uppercase string columns
        list: list of columns converted
    """
    uppercase_cols = []
    changed_count = 0

    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].str.upper().equals(df[col]):  # check if already uppercase
                continue
            df[col] = df[col].str.strip().str.upper()
            uppercase_cols.append(col)
            changed_count += 1

            if print_unique:
                print(f"{col}: {df[col].unique()}")

    print(f"\n✅ Converted {changed_count} object/string columns to uppercase out of {len(uppercase_cols)} total string columns.")
    return df, uppercase_cols

df_case_fixed, _= uppercase_all_object_columns(df)


def impute_numeric_columns(df, strategy="mean", cols=None, print_info=False, knn_neighbors=5):
    """
    Imputes missing values in numeric columns using mean, median, mode, or KNN.
    
    Parameters:
        df (pd.DataFrame): input dataframe
        strategy (str): "mean", "median", "mode", or "knn"
        cols (list or None): list of columns to impute. If None, all numeric cols are used.
        print_info (bool): if True, print info about imputations
        knn_neighbors (int): number of neighbors for KNN imputation
    Returns:
        pd.DataFrame: dataframe with imputed numeric columns
        dict: mapping of column -> imputation value used (only for mean/median/mode)
    """
    if strategy not in ["mean", "median", "mode", "knn"]:
        raise ValueError("strategy must be 'mean', 'median', 'mode', or 'knn'")

    imputed_values = {}
    changed_count = 0

    # Select numeric columns if none provided
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if strategy == "knn":
        # Select numeric columns with at least one non-NaN value
        numeric_cols = [c for c in cols if df[c].dtype in [np.float64, np.int64] and not df[c].isna().all()]
        numeric_df = df[numeric_cols].copy()

        # Apply KNN
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        imputed_array = imputer.fit_transform(numeric_df)

        # Assign back to the DataFrame
        df[numeric_cols] = pd.DataFrame(imputed_array, columns=numeric_cols, index=df.index)

        if print_info:
            print(f"\n✅ KNN imputation applied to {len(numeric_cols)} numeric columns (skipped {len(cols) - len(numeric_cols)} columns).")
        return df, {}

    # Otherwise: mean, median, or mode
    for col in cols:
        if df[col].isna().all():  # skip fully empty columns
            if print_info:
                print(f"⚠️ Skipping column '{col}' (all values are NaN)")
            continue

        if strategy == "mean":
            value = df[col].mean()
        elif strategy == "median":
            value = df[col].median()
        else:  # mode
            value = df[col].mode()[0]

        if df[col].isna().any():
            changed_count += 1

        df[col] = df[col].fillna(value)
        imputed_values[col] = value

        if print_info:
            print(f"Imputed column '{col}' with value: {value}")

    print(f"\n✅ Imputed missing values in {changed_count} columns out of {len(cols)} numeric columns.")
    return df, imputed_values


df_imputed, _ = impute_numeric_columns(df_case_fixed, strategy="knn", print_info=True)


def impute_categorical_columns(df, print_info=False):
    """
    Imputes missing values in categorical/object columns using mode (most frequent value).
    
    Parameters:
        df (pd.DataFrame): input dataframe
        print_info (bool): print info about imputations
    
    Returns:
        pd.DataFrame: dataframe with imputed categorical columns
    """
    cat_cols = df.select_dtypes(include=["object"]).columns
    changed_count = 0
    
    for col in cat_cols:
        if df[col].isna().any():
            changed_count += 1
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            if print_info:
                print(f"Imputed column '{col}' with mode: {mode_val}")  
    print(f"\n✅ Imputed missing values in {changed_count} categorical columns out of {len(cat_cols)} object columns.")
    return df

df_imputed_cat= impute_categorical_columns(df_imputed, print_info=False)