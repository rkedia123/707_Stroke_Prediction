# File Purpose: Cleaning raw data (\data\raw\subjects.csv)
# Author: Max Freitas
# Last Updated: September 29, 2025

import pandas as pd
import numpy as np
from pathlib import Path
import re
# this is needed to enable experimental features of sklearn
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge


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


def impute_numeric_columns(df, strategy="mean", cols=None, print_info=False, knn_neighbors=5, max_iter=1):
    """
    Imputes missing values in numeric columns using mean, median, mode, KNN, or regression.

    Parameters:
        df (pd.DataFrame): input dataframe
        strategy (str): "mean", "median", "mode", "knn", or "regression"
        cols (list or None): list of columns to impute. If None, all numeric cols are used.
        print_info (bool): if True, print info about imputations
        knn_neighbors (int): number of neighbors for KNN imputation
        max_iter (int): how many times to run the regression model

    Returns:
        pd.DataFrame: dataframe with imputed numeric columns
        dict: mapping of column -> imputation value used (only for mean/median/mode)
    """
    if strategy not in ["mean", "median", "mode", "knn", "regression"]:
        raise ValueError("strategy must be 'mean', 'median', 'mode', 'knn', or 'regression'")

    imputed_values = {}
    changed_count = 0

    # Select numeric columns if none provided
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_cols = [c for c in cols if df[c].dtype in [np.float64, np.int64] and not df[c].isna().all()]
    numeric_df = df[numeric_cols].copy()

    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        df[numeric_cols] = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_cols, index=df.index)
        if print_info:
            print(f"\n✅ KNN imputation applied to {len(numeric_cols)} numeric columns "
                  f"(skipped {len(cols) - len(numeric_cols)} columns).")
        return df, {}

    elif strategy == "regression":
        # IterativeImputer uses regression iteratively to fill missing values using bayesian ridge
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter, random_state=0)
        df[numeric_cols] = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_cols, index=df.index)
        if print_info:
            print(f"\n✅ Regression imputation applied to {len(numeric_cols)} numeric columns "
                  f"(skipped {len(cols) - len(numeric_cols)} columns).")
        return df, {}

    # Otherwise: mean, median, or mode
    for col in numeric_cols:
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

    if print_info:
        print(f"\n✅ Imputed missing values in {changed_count} columns out of {len(numeric_cols)} numeric columns using method: {strategy}.")
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


def clean_column_name(col):
    """
    Robust normalization of column names:
    - lowercase
    - strip whitespace and non-breaking spaces
    - replace spaces, hyphens, slashes with underscores
    - replace % with 'pct'
    - remove parentheses and other punctuation
    - collapse multiple underscores
    """
    col = col.lower()
    col = col.replace('\xa0', '_')  # non-breaking space
    col = col.strip()
    col = col.replace(' ', '_').replace('-', '_').replace('/', '_').replace('%', 'pct')
    col = re.sub(r'[()°]', '', col)  # remove parentheses and degree symbol
    col = re.sub(r'[^a-z0-9_]', '', col)  # remove any other non-alphanumeric
    col = re.sub(r'__+', '_', col)  # collapse multiple underscores
    return col


def load_data(path, selected_cols=None):
    """
    Load raw subject data into a pandas DataFrame, optionally selecting columns
    after normalizing column names for matching. Prints detailed mismatch info.
    """
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Clean CSV column names
        df.columns = [clean_column_name(c) for c in df.columns]

        if selected_cols is not None:
            sel_cols = [clean_column_name(c) for c in selected_cols]
            existing_cols = df.columns.intersection(sel_cols)
            missing_cols = set(sel_cols) - set(existing_cols)

            df = df.loc[:, existing_cols]

            print(f"📊 Selected {len(existing_cols)} matching columns.")
            if missing_cols:
                print(f"⚠️ {len(missing_cols)} columns missing:")
                for col in sorted(missing_cols):
                    print(f"   - {col}")
                
                # Optionally, show which requested columns actually exist
                print(f"✅ {len(existing_cols)} columns found:")
                for col in sorted(existing_cols):
                    print(f"   - {col}")

        return df

    except FileNotFoundError:
        print(f"❌ File not found at {path}")
        return None




##########################################################
# Testing functions and creating updated .csv file
# Define project root and data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "subjects.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "subjects_processed.csv"

cols_to_select = [
    "dm_non-dm_stroke",
    "stroke_patient_medical_history",
    "baseline_mean_hr_bp_baseline",
    "dbp_baseline",
    "mbp",
    "sbp",
    "rr_interval",
    "hr",
    "sbp_baseline",
    "hr_baseline",
    "baseline_mean_sbp_bp_baseline",
    "baseline_mean_dbp_bp_baseline",
    "baseline_mean_mbp_bp_baseline",
    "hyper_mn_hr_supine_tilt",
    "hypo_mn_hr_supine_tilt",
    "hr_supine_tilt",
    "sbp_supine_tilt",
    "dbp_supine_tilt",
    "mbp_supine_tilt",
    "hr_stand_tilt",
    "sbp_stand_tilt",
    "dbp_stand_tilt",
    "mbp_stand_tilt",
    "hrv_sdnn",
    "hrv_rmssd",
    "hrv_pnn50",
    "hrv_lf",
    "hrv_hf",
    "hrv_lf_hf",
    "rr_resp_rate",
    "o2",
    "co2",
    "o2_base_tilt",
    "o2_base_tilt_pct",
    "r2_o2_base_tilt_pct",
    "p_o2_base_tilt_pct",
    "co2_base_tilt",
    "co2_base_tilt_pct",
    "r2_co2_base_tilt_pct",
    "p_co2_base_tilt_pct",
    "rco2l_base_tilt",
    "rco2l_base_tilt_pct",
    "r2_rco2l_base_tilt_pct",
    "p_rco2l_base_tilt_pct",
    "o2_hyper_pct",
    "co2_hyper_pct",
    "o2_hypo_pct",
    "co2_hypo_pct",
    "sbp_supine_rebreathing",
    "dbp_supine_rebreathing",
    "mbp_supine_rebreathing",
    "hypo_mn_hr_supine_rebreathing",
    "hr_supine_rebreathing",
    "sbp_stand_rebreathing",
    "dbp_stand_rebreathing",
    "mbp_stand_rebreathing",
    "hr_stand_rebreathing",
    "sbp_base_tilt",
    "sbp_base_tilt_pct",
    "r2_sbp_base_tilt_pct",
    "p_sbp_base_tilt_pct",
    "dbp_base_tilt",
    "dbp_base_tilt_pct",
    "r2_dbp_base_tilt_pct",
    "p_dbp_base_tilt_pct",
    "mbp_base_tilt",
    "mbp_base_tilt_pct",
    "r2_mbp_base_tilt_pct",
    "p_mbp_base_tilt_pct",
    "sbp_hyper_pct",
    "mbp_hyper_pct",
    "mbp_hypo_pct",
    "diast_mcal_tilt",
    "mean_mcal_tilt",
    "gait_cadence",
    "gait_velocity",
    "gait_stride_time",
    "gait_stance_time",
    "htn_patient_medical_history",
    # co-variates
    "group",
    "ethnicity",
    "gender",
    "race",
    "age",
    "bmi",
    # id
    "subject_number"
]



# Load data
df = load_data(path=DATA_PATH, selected_cols= cols_to_select)

# Step 1: Convert all object/string columns to uppercase
df_case_fixed, _ = uppercase_all_object_columns(df)

#####################
#  WARNING: This can take a while to run, especially for higher max_iter
# Step 2: Impute numeric columns (regression):
df_imputed, _ = impute_numeric_columns(df_case_fixed, strategy="regression", print_info=True, max_iter=5)

# Step 3: Impute categorical/object columns (hot-deck imputation)
# using stratifiers: `group`, `gender`, `ethnicity`, `race`
strat_cols=['group', 'gender', 'ethnicity', 'race']
df_imputed_cat = impute_categorical_columns(df_imputed, print_info=False, strategy= "hot_deck", stratify_cols=strat_cols)

# Step 4: Save the processed DataFrame to CSV
# Ensure the output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_imputed_cat.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Processed data saved to: {OUTPUT_PATH}")
