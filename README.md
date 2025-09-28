# 707_Stroke_Prediction
Determining Key Biomarkers of Stroke in ICU Patients
---
### 1. Data Processing Steps

- **Object/String Capitalization**  
  Standardizes all object/string columns by converting values to uppercase using  `uppercase_all_object_columns`

- **Numeric Column Imputation**  
  Handles missing values in numeric columns using `impute_numeric_columns`
  
  Supported strategies:  
  * **Mean imputation** – replaces missing values with the column mean  
  * **Median imputation** – replaces missing values with the column median 
  * **Mode imputation** – replaces missing values with the most frequent value 
  * **KNN imputation** – replaces missing values using K-Nearest Neighbors

- **Categorical Column Imputation**  
  Handles missing values in categorical/object columns using `impute_categorical_columns`
  * **Mode imputation** – replaces missing values with the most frequent category in the column
