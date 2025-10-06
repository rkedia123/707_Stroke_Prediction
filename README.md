# 707_Stroke_Prediction
Determining Key Biomarkers of Stroke in ICU Patients
---
### 1. Data Processing Steps

- **Load Data**
  * Load in raw data and select cols of interest using `load_data`

- **Object/String Capitalization**  
  * Standardizes all object/string columns by converting values to uppercase using  `uppercase_all_object_columns`

- **Numeric Column Imputation**  
  * Handles missing values in numeric columns using `impute_numeric_columns`
  
  Supported strategies:  
  * **Mean imputation** – replaces missing values with the column mean  
  * **Median imputation** – replaces missing values with the column median 
  * **Mode imputation** – replaces missing values with the most frequent value 
  * **KNN imputation** – replaces missing values using K-Nearest Neighbors
  * **Regression imputation**: replace missing values using a regression model
    - Each missing column is predicted using `Bayesian Ridge Regression` with other numeric columns as predictors
    - Why `Bayesian Ridge`: regularizes automatically via a prior, for a more stable imputation than using a linear regression. Provides a good balance of bias and variance.
   

- **Categorical Column Imputation**  
  Handles missing values in categorical/object columns using `impute_categorical_columns`
  * **Mode imputation** – replaces missing values with the most frequent category in the column
  * **Hot Deck imputation** - replaces missing values based on values, where the patients having similar stratification columns
    - Step 1: Identify catgorical coluns with missing values
    - Step 2: For each missing value, decide which donor pool to use
      * If stratfiers are provided, use values from the same group (stratified)
      * If no stratifiers are provided, use all observed values in the column (global)
    - Step 3: Replace each missing value with a randomly selected donor 
    - Potential list of stratifiers (`group`, `gender`, `ethnicity`, `race`). It's best to only use a few stratifiers otherwise, we will have to use global column set too much.
