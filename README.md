# Battery State of Health (SoH) Prediction

Predicts Li-ion battery **State of Health** (remaining capacity as a fraction of original capacity) from charge/discharge cycling data and electrochemical impedance spectroscopy (EIS) measurements, using the NASA battery dataset.

## Data

- **Source:** [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) (Kaggle, via `kagglehub`), a cleaned/CSV version of NASA's original MATLAB battery cycling data.
- **Contents:** charge cycles, discharge cycles, and EIS (impedance) cycles for ~50 Li-ion cells run under different discharge profiles (constant 1A/2A/4A, alternating, square-wave) and ambient temperatures.
- **Target:** `SoH = Capacity / max(Capacity)` per cell.

## Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
plotly
scikit-learn
xgboost
lightgbm
tensorflow
kagglehub
impedance
tqdm
prettytable<3.10
```

A Kaggle account/API access is needed for `kagglehub` to download the dataset on first run.

## Notebook structure

1. **Setup & Imports** — install/import all libraries used throughout.
2. **Load & Parse Raw NASA Battery Data** — download the dataset, parse MATLAB-style timestamps, build `charge_df`, `discharge_df`, `eis_df`.
3. **Decompose Complex EIS Values** — split complex impedance/current columns into real/imaginary components.
4. **Label Discharge Type & Compute SoH** — tag each row with its discharge profile and compute per-cell SoH.
5. **EDA: SoH Trends by Discharge Type** — plot SoH vs. cycle number to spot abnormal degradation.
6. **Remove Problem Cells** — drop cells with premature failure or anomalous degradation.
7. **Re-check Cleaned Data** — confirm the cleaned dataset looks reasonable.
8. **Additional EDA** — sanity checks on `discharge_df`.
9. **Aggregate to Per-Cycle Summary Tables** — collapse raw time-series rows into one row per (Cell_ID, Cycle_Number).
10. **Merge EIS Features into Charge-Discharge Summary** — align EIS cycles (±1) with the nearest discharge cycle and merge.
11. **Correlation Analysis: With vs. Without EIS Features** — check whether EIS-derived resistances add signal for predicting SoH.
12. **Preprocess for Modeling** — one-hot encode categoricals, split train/test **by Cell_ID** to prevent data leakage.
13. **Define Model Pipelines, CV Strategy & Hyperparameter Grids** — Random Forest / XGBoost / LightGBM pipelines with 5-fold group-aware CV (`GroupKFold`).
14. **Train & Evaluate Models** — grid search and evaluate each model (14.1–14.3), then compare results (14.4).
15. **Feature Importance of the Three Models** — compare top features driving each model's predictions.
16. **Conclusion** — summary of findings.

## Results

| Model | R² | MSE | RMSE | MAE |
|---|---|---|---|---|
| Random Forest | 0.66 | 0.0043 | 0.0655 | 0.0504 |
| XGBoost | 0.64 | 0.0045 | 0.0673 | 0.0515 |
| **LightGBM** | **0.78** | **0.0027** | **0.0524** | **0.0387** |

**LightGBM** was the best-performing model across every metric after group-aware hyperparameter tuning.

**`Voltage_measured`** was the most important feature across all three models, consistent with the correlation analysis showing measured/load voltage strongly associated with SoH.

## Notes

- Train/test splits are done **by Cell_ID**, not by row, so no cycles from the same battery leak across the split.
- Cross-validation during grid search uses `GroupKFold` (grouped by Cell_ID) for the same reason — model selection never sees validation folds contaminated by cells it trained on.
