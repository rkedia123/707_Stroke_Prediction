# File Purpose: Cleaning raw data (\data\raw\subjects.csv)
# Author: Max Freitas
# Last Updated: September 28, 2025

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer


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
            if df[col].str.upper().equals(df[col]):
                continue
            df[col] = df[col].str.strip().str.upper()
            uppercase_cols.append(col)
            changed_count += 1

            if print_unique:
                print(f"{col}: {df[col].unique()}")

    print(f"\n✅ Converted {changed_count} object/string columns to uppercase "
          f"out of {len(uppercase_cols)} total string columns.")
    return df, uppercase_cols


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
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        imputed_array = imputer.fit_transform(numeric_df)

        # Assign back to the DataFrame
        df[numeric_cols] = pd.DataFrame(imputed_array, columns=numeric_cols, index=df.index)

        if print_info:
            print(f"\n✅ KNN imputation applied to {len(numeric_cols)} numeric columns "
                  f"(skipped {len(cols) - len(numeric_cols)} columns).")
        return df, {}

    # Otherwise: mean, median, or mode
    for col in cols:
        if df[col].isna().all():
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

    print(f"\n✅ Imputed missing values in {changed_count} columns out of {len(cols)} numeric columns using method: {strategy}.")
    return df, imputed_values


def impute_categorical_columns(df, stratify_cols=None, print_info=False, strategy="mode"):
    """
    Imputes missing values in categorical/object columns using mode or hot-deck. 

    Parameters:
        df (pd.DataFrame): input dataframe
        stratify_cols (list[str], optional): columns to use for conditional hot-deck. If none, use global cols
        print_info (bool): print info about imputations
        strategy (str): "mode" or "hot_deck"

    Returns:
        pd.DataFrame: dataframe with imputed categorical columns
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    changed_count = 0

    if strategy == "hot_deck":
        for col in cat_cols:
            if df[col].isna().any():
                changed_count += 1
                global_donors = df[col].dropna().values  # fallback pool
                stratified_count = 0
                fallback_count = 0

                if stratify_cols:
                    grouped = df.groupby(stratify_cols)
                    for keys, subdf in grouped:
                        observed = subdf[col].dropna().values
                        if len(observed) == 0:
                            # fallback to global pool
                            df.loc[subdf.index, col] = subdf[col].apply(
                                lambda x: np.random.choice(global_donors) if pd.isna(x) else x
                            )
                            fallback_count += subdf[col].isna().sum()
                        else:
                            df.loc[subdf.index, col] = subdf[col].apply(
                                lambda x: np.random.choice(observed) if pd.isna(x) else x
                            )
                            stratified_count += subdf[col].isna().sum()
                else:
                    # purely random hot-deck
                    df[col] = df[col].apply(
                        lambda x: np.random.choice(global_donors) if pd.isna(x) else x
                    )
                    fallback_count += df[col].isna().sum()  # all missing are global

                if print_info:
                    print(f"Imputed column '{col}' using hot-deck")
                    if stratify_cols:
                        print(f"  → {stratified_count} values imputed using stratification")
                        print(f"  → {fallback_count} values imputed using global fallback")
                    else:
                        print(f"  → {fallback_count} values imputed using global pool (no stratification)")

    elif strategy == "mode":
        for col in cat_cols:
            if df[col].isna().any():
                changed_count += 1
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                if print_info:
                    print(f"Imputed column '{col}' with mode: {mode_val}")

    print(f"\n✅ Imputed missing values in {changed_count} categorical columns "
          f"out of {len(cat_cols)} object columns using method: {strategy}.")
    return df



##########################################################
# Testing functions and creating updated .csv file
# Define project root and data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "subjects.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "subjects_processed.csv"

# Function to load raw data
def load_data(path=DATA_PATH):
    """Load raw subject data into a pandas DataFrame."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found at {path}")
        return None

# Load data
df = load_data()

# Step 1: Convert all object/string columns to uppercase
df_case_fixed, _ = uppercase_all_object_columns(df)

# Step 2: Impute numeric columns (KNN imputation)
df_imputed, _ = impute_numeric_columns(df_case_fixed, strategy="knn", print_info=True)

# Step 3: Impute categorical/object columns (hot-deck imputation)
# using stratifiers: `group`, `gender`, `ethnicity`, `race`
strat_cols=['group', 'gender', 'ethnicity', 'race']
df_imputed_cat = impute_categorical_columns(df_imputed, print_info=True, strategy= "hot_deck", stratify_cols=strat_cols)

# Step 4: Save the processed DataFrame to CSV
# Ensure the output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_imputed_cat.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Processed data saved to: {OUTPUT_PATH}")
