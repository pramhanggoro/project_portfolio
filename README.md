# Battery State of Health (SoH) Prediction

Predicts Li-ion battery **State of Health** (remaining capacity as a fraction of original capacity) from discharge-cycle summaries and electrochemical impedance spectroscopy (EIS) measurements, using the NASA battery dataset.

## Data

- **Source:** [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) (Kaggle, via `kagglehub`), a cleaned/CSV version of NASA's original MATLAB battery cycling data (B. Saha and K. Goebel, 2007, NASA Prognostics Data Repository, NASA Ames Research Center).
- **Contents:** charge cycles, discharge cycles, and EIS (impedance) cycles for 34 Li-ion cells run under different discharge profiles (constant 1A/2A/4A, alternating loads, square-wave) and ambient temperatures (4 °C, 24 °C, 43 °C).
- **Target:** `SoH = Capacity / max(Capacity)` per cell.
- **Modeling set:** after removing problem cells and keeping only discharge cycles that sit next to an EIS measurement, 479 rows remain (199 train / 280 test).

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
kagglehub
pyarrow<21
```

- A Kaggle account/API access is needed for `kagglehub` to download the dataset on first run.
- `pyarrow<21` is only needed in notebook 01, to save the dataframes as pickles.

## Notebooks

Run the notebooks in order. Each one saves pickles that the next one loads from the working directory.

| Notebook | Purpose | Reads | Writes |
|---|---|---|---|
| `01_SoH_NASA_LIB_Extraction.ipynb` | Download and parse the raw data | Kaggle dataset | `charge_df.pkl`, `discharge_df.pkl`, `eis_df.pkl` |
| `02_SoH_NASA_LIB_EDA.ipynb` | Clean, label, compute SoH, build the modeling table | the three pickles above | `charge_df_EDA.pkl`, `discharge_df_EDA.pkl`, `eis_df_EDA.pkl`, `discharge_eis_clean.pkl` |
| `03_SOH_NASA_LIB_ML.ipynb` | Train and compare models | the `*_EDA.pkl` files and `discharge_eis_clean.pkl` | – |

### 01 – Extraction
1. **Setup & Imports**
2. **Load & Parse Raw NASA Battery Data**: download the dataset, parse MATLAB-style timestamps, build `charge_df`, `discharge_df`, `eis_df`, drop cells B0049–B0052 (control-software crashes), and merge in a per-cell protocol table (discharge mode, current, ambient temperature, cut-off voltage, end-of-life criterion).
3. **Installing pyarrow** to allow saving dataframes as pickles.

### 02 – EDA
1. **Importing Libraries**
2. **Decompose Complex EIS Values**: split complex impedance/current columns into real/imaginary parts and add |Z|.
3. **Label Discharge Type & Compute SoH**
4. **EDA: SoH Trends by Discharge Type**: plot SoH vs. cycle number to spot abnormal degradation.
5. **Remove Problem Cells**: drop the 43 °C batch (B0029–B0032) and cells with premature or anomalous failure (B0049–B0054).
6. **Re-check Cleaned Data**
7. **Additional EDA**
8. **Aggregate to Per-Cycle Summary Tables**: one row per (Cell_ID, Cycle_Number).
9. **Merge EIS Features into Charge-Discharge Summary**: shift EIS cycle numbers by ±1 to match the discharge cycle before/after each impedance scan, then drop rows without EIS data.
10. **Saving post-EDA dataframes**

### 03 – Machine Learning
1. **Setup & Imports**
2. **Correlation Analysis**: check whether the EIS-derived resistances carry signal for SoH.
3. **Preprocess for Modeling**: one-hot encode `Discharge_Type`, split train/test **by Cell_ID** (`Cell_ID` itself is not a feature).
4. **Define Model Pipelines, CV Strategy & Hyperparameter Grids**: Random Forest / XGBoost / LightGBM with 5-fold `GroupKFold`.
5. **Train & Evaluate Models**: grid search and test-set evaluation per model (5.1–5.3), comparison (5.4), feature importance (5.5), predicted-vs-actual plots (5.6).
6. **Further Comments**

## Features

17 features per cycle: cycle number, discharge duration, discharge current, voltage drop, minimum current, temperature rise, ambient temperature, mean load voltage/current, cut-off voltage, one-hot discharge type, and four EIS resistances (`re_dis`, `rct_dis`, `re_charge`, `rct_charge`: electrolyte and charge-transfer resistance from the impedance scans before and after the discharge cycle).

## Results

Evaluated on held-out test cells:

| Model | R² | MSE | RMSE | MAE |
|---|---|---|---|---|
| Random Forest | 0.70 | 0.0038 | 0.0613 | 0.0479 |
| XGBoost | 0.42 | 0.0074 | 0.0858 | 0.0681 |
| **LightGBM** | **0.78** | **0.0027** | **0.0524** | **0.0387** |

**LightGBM** was the best-performing model across every metric after hyperparameter tuning.

- **Feature importance:** for LightGBM, the electrolyte resistance after a discharge cycle (`re_dis`) is the most important feature. For XGBoost, it is the charge-transfer resistance after a charging cycle (`rct_charge`). For Random Forest, it is the experiment duration.
- **Where the models fail:** all three perform reasonably up to an SoH of about 0.8. XGBoost is the weakest and often predicts about 0.8 when the actual SoH is much lower. Random Forest starts to deviate while the actual SoH is still above 0.7. LightGBM tracks well down to about 0.7.

## Notes

- Train/test splits are done **by Cell_ID**, not by row, so no cycles from the same battery leak across the split. `Cell_ID` is used only for grouping, never as a feature.
- Cross-validation during grid search uses `GroupKFold` (grouped by Cell_ID) for the same reason.
- The modeling set is small (479 rows from a limited number of cells) and the test set contains only a few cells, so the metrics are noisy and depend on which cells fall in the test split.

## Limitations & Future Work

Equivalent-circuit-model (ECM) fitting of the EIS spectra was attempted to extract more features (double-layer capacitance, diffusion elements, etc.), but it was not pursued. The EIS data in the dataset has large Kramers-Kronig residuals at the low- and high-frequency ends, and removing the problematic points would discard almost half of each spectrum. The dataset also doesn't document how the EIS data was fitted or whether any points were omitted. Only the electrolyte resistance (Re) and charge-transfer resistance (Rct) supplied with the dataset are used as EIS features.
