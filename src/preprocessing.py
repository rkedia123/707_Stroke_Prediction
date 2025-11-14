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

def clean_column_list(col_list):
    """
    Apply clean_column_name to a list of column names.

    Parameters:
        col_list (list): list of string column names

    Returns:
        list: list of cleaned column names
    """
    return [clean_column_name(col) for col in col_list]

def load_data(path, selected_cols=None):
    """
    Load raw subject data into a pandas DataFrame, optionally selecting columns
    after normalizing column names for matching. Prints detailed mismatch info.

    Parameters:
        path (str): path to CSV file
        selected_cols (list or None): list of columns to select after cleaning names

    Returns:
        pd.DataFrame or None: dataframe with loaded (and optionally selected) columns,
                              or None if file not found
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







##########################################################
# Testing functions and creating updated .csv file
# Define project root and data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "subjects.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "subjects_processed.csv"
SUBSET_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "subset_subjects_processed.csv"

# Demographics & Clinical
demographics_clinical = [
    "subject_number",
    "Age",
    "gender",
    "BMI",
    "Group",
    "ethnicity",
    "race",
    "DM PATIENT MEDICAL HISTORY",
    "HTN YRS PATIENT MEDICAL HISTORY"
]

# Baseline Cardiovascular
baseline_cardiovascular = [
    "(Baseline Mean) HR BP BASELINE",
    "SBP BASELINE",
    "DBP BASELINE",
    "MBP",
    "MEAN MCAR BASELINE",
    "MEAN MCAL BASELINE",
    "CO2 BASELINE",
    "SYST MCAR BASELINE",
    "DIAST MCAR BASELINE"
]

# Tilt (Autonomic / Orthostatic)
tilt_autonomic = [
    "(Tilt mn) HR TILT",
    "SBP TILT",
    "DBP TILT",
    "MBP TILT",
    "MEAN MCAR TILT",
    "MEAN MCAL TILT",
    "CO2 TILT",
    "DELTA MEAN BP TILT-BASELINE"
]

# Hyperventilation
hyperventilation = [
    "(Hyper mn) HR HYPERVENTILATION",
    "SBP HV",
    "DBP HV",
    "MBP HV",
    "MEAN MCAR HV",
    "MEAN MCAL HV",
    "CO2 HV",
    "CO2_reactivity_Baseline-2-Hyper_MCAR",
    "CO2_reactivity_Baseline-2-Hyper_MCAL"
]

# Rebreathing
rebreathing = [
    "(Hypo mn) HR SUPINE REBREATHING",
    "SBP SUPINE REBREATHING",
    "DBP SUPINE REBREATHING",
    "MBP SUPINE REBREATHING",
    "MEAN MCAR SUPINE REBREATHING",
    "MEAN MCAL SUPINE REBREATHING",
    "CO2 SUPINE REBREATHING",
    "CO2_reactivity_Baseline-2-Hypo_MCAR",
    "CO2_reactivity_Baseline-2-Hypo_MCAL"
]

# Sitting & Standing Responses
sitting_standing = [
    "(SitEO mn) SitEO HR mean",
    "Mean BP SitEO",
    "Mean MCAR SitEO",
    "Mean MCAL SitEO",
    "(StandEO mn) Mean HR StandEO",
    "Mean BP StandEO",
    "Mean MCAR StandEO",
    "Mean MCAL StandEO"
]

# Cerebrovascular Age-Adjusted Indices
cerebrovascular_indices = [
    "Age_adjusted_Mean_MCAR_BASELINE",
    "Age_adjusted_Mean_MCAL_BASELINE",
    "Age_Residual MEAN MCAR BASELINE",
    "Age_Residual MEAN MCAL BASELINE",
    "CO2_reactivity_MCAR"
]

# 24-Hour Blood Pressure Variability
bp_variability_24hr = [
    "24Hour-Daytime-SBP",
    "24Hour-Nighttime-SBP",
    "24Hour-Daytime-DBP",
    "24Hour-Nighttime-DBP",
    "24Hour-MBPDIP%",
    "24Hour-HRDIP%",
    "24Hour-SBPDIPPER",
    "24Hour-DBPDIPPER",
    "24Hour-MBPDIPPER"
]

# Heart Rate Variability Indices
hr_variability = [
    "HRV_ SDNN",
    "HRV_ RMSSD",
    "HRV_PNN50",
    "HRV_ LF",
    "HRV_ HF",
    "HRV_ LF/HF",
    "HRV_ AVNN",
    "HRV_ SDANN"
]

# Gas Exchange & Vasoreactivity
gas_exchange_vasoreactivity = [
    "O2_base",
    "CO2_base",
    "O2_hyper_%",
    "CO2_hyper_%",
    "O2_hypo_%",
    "CO2_hypo_%",
    "O2_tilt_%",
    "CO2_tilt_%"
]

# Gait & Movement
gait_movement = [
    "GAIT - Walk 1 distance (m)",
    "GAIT - Walk 1 speed (m/s)",
    "GAIT - Walk 2 distance (m)",
    "GAIT - Walk 2 speed (m/s)"
]

all_columns = (
    demographics_clinical +
    baseline_cardiovascular +
    tilt_autonomic +
    hyperventilation +
    rebreathing +
    sitting_standing +
    cerebrovascular_indices +
    bp_variability_24hr +
    hr_variability +
    gas_exchange_vasoreactivity +
    gait_movement
)

# Apply cleaning function to all column names
cleaned_columns = [clean_column_name(col) for col in all_columns]


# Load data
df = load_data(path=DATA_PATH, selected_cols= cleaned_columns)

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

# Step 5: Save subset data based off corr
SUBSET_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# cols to use
cols=[
    "gait_walk_1_distance_m",  # gait
    "o2_base", # gas exchange
    "hrv_sdnn", # heart rate
    "24hour_mbpdippct", #blood pressure
    "age_adjusted_mean_mcar_baseline", #cerebrovascular 
    "mean_bp_siteo", # sitting/standing
    "mean_mcar_supine_rebreathing", # rebreathing
    "sbp_hv", # hyperventilation
    "mean_mcar_tilt", # tilt
    "baseline_mean_hr_bp_baseline", #baseline cardiovascular
    "age", # demographics
    "bmi", # demographics
    "htn_yrs_patient_medical_history", # demographics
    "group", # demographics
    "gender", # demographics
    "ethnicity", # demographics
    "race", # demographics
    "dm_patient_medical_history", # demographics
    "24hour_sbpdipper" # demographics

]
df_imputed_subset= df_imputed_cat[cols]
df_imputed_subset.to_csv(SUBSET_OUTPUT_PATH, index=False)

print(f"✅ Processed subsetted data saved to: {SUBSET_OUTPUT_PATH}")
