# Unicorn Wealth - Feature Implementation Guide for AI Agent

This document is the definitive guide for implementing features from the `Unicorn_Wealth_Feature_Set.json` blueprint. You must read and adhere to these rules for every feature implementation task. The goal is to translate the JSON specification into hardcoded, tested, and permanent Python code. The JSON file itself must **not** be read by the application at runtime.
### IMPORTANT NOTE: ALL data rows in the database tables feature_store_1h, feature_store_4h & feature_store_8h are 15m rows of data. The time appendixes in the title of the tables are references to time horizons for target setting only. All features (is_ml_training_feature=TRUE in unicorn_wealth_feature_set.json), should be saved to ALL 3 feature_store tables, with the only difference being the settings of some indicators as outlined by FEATURE_PARAMS in config.py
---
### Variable Interpretation
- **`{t}`:** This is a placeholder for the token symbol (e.g., 'BTC', 'ETH'). The `UnifiedFeatureEngine` must replace this with the actual token ID from the `MASTER_TOKEN_LIST` when calling the feature function.

---
### Key Interpretation and Action Framework

| JSON Key | Intent & Required Action |
| :--- | :--- |
| **Core Definition** | |
| `operation` | **The Function Name.** The Python function you create must be named exactly this (e.g., `adx_15m`). |
| `live_cadence` | **The Execution Schedule.** Defines when the feature is calculated live. Your function will be called by the hardcoded pipeline in the `UnifiedFeatureEngine` corresponding to this cadence (e.g., `run_15m_pipeline`). |
| `step_order` | **Execution Order.** Critical. Your call to the new function inside the engine's hardcoded pipeline method must be placed correctly according to this number, respecting all data dependencies. |
| `calculate_per_token` | **Looping Logic.** If `true`, the `UnifiedFeatureEngine` must loop through the `MASTER_TOKEN_LIST` and call this function for each token. If `false`, the function is called only once per cycle. |
| **Transformation** | |
| `transform_data_source` | **Function Inputs.** The names of the source DataFrames the function depends on. The Python function signature must accept these as arguments. |
| `transform` | **The Core Logic.** This is the specific business logic that must be implemented inside the Python function. |
| `output_data_type` | **Data Type.** The pandas/numpy data type of the output (e.g., `float64`, `int32`). The function should ensure its output conforms to this type. |
| **Data Storage & Persistence** | |
| `df_variable_id` | **Registry Key.** If this feature's output is an intermediate DataFrame needed by other features, this is the unique key used to store it in the `DataFrameRegistry`. |
| `df_storage_period` | **Memory Size.** The maximum number of rows to keep for this DataFrame. This integer **must** be passed as the `storage_period` argument to the `registry.store_df` method. |
| `df_keys` | **DataFrame Columns.** If creating a new DataFrame, these are the required column names, starting with the timestamp. |
| `df_frame_store` | **Storage Flag.** If `TRUE`, the output of this function is a foundational DataFrame that must be stored in the registry. |
| `df_update_mode` | **Storage Method.** The method for storing the DataFrame (e.g., `rolling_append`). This string **must** be passed as the `update_mode` argument to `registry.store_df`. |
| `is_historical_export_enabled`| **Historical Save Flag.** If `true`, the raw data fetched by this operation must be saved to the database during the `run-historical` pipeline. This provides the long-term data for model training. |
| `is_live_append_enabled`| **Live Save Flag.** If `true`, the new data fetched by this operation during its live_cadence must also be saved to the database (rolling append) to fill gaps for future training runs. |
| **ML & Data Quality** | |
| `nan_handing` | **Imputation Strategy.** Defines how to handle missing values (NaNs) in the output of this feature. This logic must be implemented within the function. |
| `is_categorical_feature`| **Categorical Flag.** If `true`, this feature must be treated as a categorical variable by the ML data preparation pipeline. |
| `is_ml_training_feature`| **Training Flag.** If `true`, this feature is used when training models. |
| `ml_feature_lag_...` | **Lagged Features.** If `true` for a specific lag (e.g., `ml_feature_lag_4`), the data preparation pipeline must create a corresponding lagged version of this feature. |
| `is_ml_live_feature` | **Live Flag.** Critical. If `true`, this feature is part of the final feature set passed to the live models for real-time inference. |
| `is_ml_target` | **Target Flag.** If `true`, this column represents the target variable (y) for model training. |
| **Pipeline Control** | |
| `on_error` | **Error Handling.** Defines the behavior if the function fails (e.g., `halt`, `retry`, `skip`). This logic must be wrapped around the function call in the `UnifiedFeatureEngine`. |
| `max_retries` | **Retry Count.** If `on_error` is `retry`, this specifies the number of retries. |
| `log_level` | **Logging Verbosity.** Specifies the log level for messages related to this specific feature's execution. |
