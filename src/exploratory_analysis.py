# File Purpose: Create tables and exploratory analysis for (\data\processed\subjects_processed.csv)
# Author: Max Freitas
# Last Updated: November 12, 2025

import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os 
from preprocessing import clean_column_name
from sklearn.preprocessing import StandardScaler


# for cat columns
def summarize_categorical(df, round_digits=1, exclude=None):
    """
    Summarize categorical columns with count and percent in a single column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    round_digits : int, optional
        Number of decimal places for percentages
    exclude : list, optional
        List of column names to exclude from the summary

    Returns
    -------
    pd.DataFrame
        Summary table with Variable | Category | Count (%)
    """
    summary_rows = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if exclude is not None:
        cat_cols = [col for col in cat_cols if col not in exclude]

    for col in cat_cols:
        series = df[col].dropna()
        counts = series.value_counts()
        percents = series.value_counts(normalize=True) * 100

        for cat, count in counts.items():
            summary_rows.append({
                "Variable": col,
                "Category": cat,
                "Count (%)": f"{count} ({percents[cat]:.{round_digits}f}%)"
            })
    
    return pd.DataFrame(summary_rows)

# numeric cols
def summarize_numeric(df, groups_dict, round_digits=1):
    """
    Summarize numeric variables by predefined groups with blank lines between each group.
    Only includes numeric columns from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    groups_dict : dict
        Dictionary mapping group name -> list of variable names.
    round_digits : int, default=1
        Number of decimal places for rounding numeric stats.

    Returns
    -------
    pd.DataFrame
        Summary table grouped by section with blank rows between groups.
    """
    summary_rows = []

    # Get all numeric columns in the dataframe
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for group_name, var_list in groups_dict.items():
        # Add group header row
        summary_rows.append({
            "Variable": f"== {group_name} ==",
            "N": "",
            "Mean (SD)": "",
            "Min": "",
            "Max": ""
        })

        # Loop through each variable in the group
        for col in var_list:
            if col not in numeric_cols:
                continue  # skip non-numeric or missing columns

            series = df[col].dropna()
            if series.empty:
                continue  # skip if no valid data

            summary_rows.append({
                "Variable": col,
                "N": series.count(),
                "Mean (SD)": f"{series.mean():.{round_digits}f} ({series.std():.{round_digits}f})",
                "Min": round(series.min(), round_digits),
                "Max": round(series.max(), round_digits)
            })

        # Add a blank separator row
        summary_rows.append({
            "Variable": "",
            "N": "",
            "Mean (SD)": "",
            "Min": "",
            "Max": ""
        })

    return pd.DataFrame(summary_rows)

# create corr heat map
def plot_numeric_corr_heatmap(
    df, 
    threshold=0.8, 
    figsize=(12,10), 
    save_heatmap_path=None, 
    save_corr_df_path=None,
    cols=None
):
    """
    Plot a heatmap of correlations between numeric columns (or selected columns)
    and flag highly correlated pairs.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
    figsize : tuple
    save_heatmap_path : str or Path
    save_corr_df_path : str or Path
    cols : list, optional
        Columns to include in the correlation heatmap. 
        If None, all numeric columns are used.
    """

    # --- Select columns ---
    if cols is not None:
        # Only keep numeric among requested cols
        numeric_df = df[cols].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns available for correlation.")

    # --- Compute correlation ---
    corr_matrix = numeric_df.corr()

    # --- Plot heatmap ---
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    if save_heatmap_path:
        plt.savefig(save_heatmap_path, dpi=300)

    plt.show()

    # --- Find highly correlated pairs ---
    high_corr = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr.append({
                    "Variable 1": cols[i],
                    "Variable 2": cols[j],
                    "Correlation": corr_val
                })

    high_corr_df = pd.DataFrame(high_corr).sort_values(
        by="Correlation", 
        key=lambda x: abs(x),
        ascending=False
    )

    # save
    if save_corr_df_path:
        high_corr_df.to_csv(save_corr_df_path, index=False)

    return high_corr_df


def plot_numeric_boxplots(df, groups, figsize=(12,6), output_dir=None, standardize=True):
    """
    Plot boxplots for numeric columns in different groups, saving separate images for each group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    groups : dict
        Dictionary where keys are group names and values are lists of column names.
    figsize : tuple, default=(12,6)
        Figure size for the plots.
    output_dir : str or Path, optional
        Directory to save the images. If None, images are not saved.

    Returns
    -------
    None
    """

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for group_name, cols in groups.items():
        # Filter numeric columns that exist in the DataFrame
        valid_cols = [col for col in cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not valid_cols:
            print(f"No numeric columns found for group '{group_name}'. Skipping.")
            continue

        df_group = df[valid_cols].copy()

        if standardize:
            scaler = StandardScaler()
            df_group_scaled = pd.DataFrame(scaler.fit_transform(df_group), columns=valid_cols)
        df_melted = df_group_scaled.melt(var_name="Variable", value_name="Value")

        # Plot boxplot
        plt.figure(figsize=figsize)
        sns.boxplot(x="Variable", y="Value", data=df_melted)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Boxplots: {group_name}")
        plt.tight_layout()

        # Save figure if output_dir provided
        if output_dir:
            safe_group_name = re.sub(r'[<>:"/\\|?*]', '_', group_name)
            filename = os.path.join(output_dir, f"{safe_group_name}_boxplot.png")
            plt.savefig(filename, dpi=300)

        plt.show()

# standard box-plot
def plot_standardized_boxplot(df, figsize=(12,6), output_path=None):
    """
    Standardize all numeric columns in the DataFrame and plot a single boxplot.
    Optionally save the figure to a file.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    figsize : tuple, default=(12,6)
        Figure size for the plot.
    output_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    None
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        print("No numeric columns found in the DataFrame.")
        return

    # Standardize numeric columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

    # Melt for plotting
    df_melted = df_scaled.melt(var_name="Variable", value_name="Value")

    # Plot boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(x="Variable", y="Value", data=df_melted, fliersize=1.5)
    plt.xticks(rotation=45, ha="right")
    plt.title("Standardized Boxplots of All Numeric Variables")
    plt.tight_layout()

    # Save figure if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()

##########################################################
# Testing functions and creating updated .csv file
# Define project root and data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "subjects_processed.csv"
SUBSET_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "subset_subjects_processed.csv"

# load data
df_full = pd.read_csv(MAIN_DATA_PATH)
df_subset = pd.read_csv(SUBSET_DATA_PATH)

# Generate summaries
full_cat_summary = summarize_categorical(df_full, exclude= "subject_number")
subset_cat_summary = summarize_categorical(df_subset, exclude= "subject_number")
# group by 

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

groups = {
    "Demographics & Clinical": demographics_clinical,
    "Baseline Cardiovascular": baseline_cardiovascular,
    "Tilt (Autonomic / Orthostatic)": tilt_autonomic,
    "Hyperventilation": hyperventilation,
    "Rebreathing": rebreathing,
    "Sitting & Standing Responses": sitting_standing,
    "Cerebrovascular Age-Adjusted Indices": cerebrovascular_indices,
    "24-Hour Blood Pressure Variability": bp_variability_24hr,
    "Heart Rate Variability Indices": hr_variability,
    "Gas Exchange & Vasoreactivity": gas_exchange_vasoreactivity,
    "Gait & Movement": gait_movement
}


# Apply cleaning function to all column names
groups_cleaned = {
    group_name: [clean_column_name(col) for col in cols]
    for group_name, cols in groups.items()
}
full_num_summary = summarize_numeric(df_full, groups_cleaned)
subset_num_summary = summarize_numeric(df_subset, groups_cleaned)

# Save outputs
full_cat_output_path = PROJECT_ROOT / "data" / "processed" / "tables"/ "full_categorical_summary.csv"
full_num_output_path = PROJECT_ROOT / "data" / "processed" / "tables" / "full_numeric_summary.csv"

subset_cat_output_path = PROJECT_ROOT / "data" / "processed" / "tables"/ "subset_categorical_summary.csv"
subset_num_output_path = PROJECT_ROOT / "data" / "processed" / "tables" / "subset_numeric_summary.csv"

full_cat_summary.to_csv(full_cat_output_path, index=False)
full_num_summary.to_csv(full_num_output_path, index=False)

subset_cat_summary.to_csv(subset_cat_output_path, index=False)
subset_num_summary.to_csv(subset_num_output_path, index=False)




#### 
# Create Plots
# Heat Map Corr Plot + csv
full_plot_path= PROJECT_ROOT/ "data"/"processed" / "plots" / "corr_heatmap_full.png"
full_corr_table_path= PROJECT_ROOT/ "data"/"processed" / "tables" / "high_corr_table_full.csv"
subset_plot_path= PROJECT_ROOT/ "data"/"processed" / "plots" / "corr_heatmap_subset.png"
subset_corr_table_path= PROJECT_ROOT/ "data"/"processed" / "tables" / "high_corr_table_subset.csv"



high_corr_df = plot_numeric_corr_heatmap(
    df_full, 
    threshold=0.5, 
    save_heatmap_path=full_plot_path,
    save_corr_df_path= full_corr_table_path,
)

high_corr_df = plot_numeric_corr_heatmap(
    df_subset, 
    threshold=0.5, 
    save_heatmap_path=subset_plot_path,
    save_corr_df_path= subset_corr_table_path,
)

# box plots
output_dir= PROJECT_ROOT/ "data"/"processed" / "plots"
plot_numeric_boxplots(df_full, output_dir= output_dir, groups= groups_cleaned)

# subset of standardized num cols
box_plot_path=  PROJECT_ROOT/ "data"/"processed" / "plots" / "subset_standard_box.png"
plot_standardized_boxplot(df_subset, output_path=box_plot_path)