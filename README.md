# NASA Lithium-ion Battery SOC Prediction

This project focuses on cleaning, preparing, and modeling NASA’s Lithium-ion Battery dataset to predict the **State of Charge (SOC)** of individual cells.  
It demonstrates practical skills in data cleaning, exploratory data analysis (EDA), feature engineering, and the application of both traditional and deep learning models.

---

## Project Overview

Lithium-ion batteries are central to modern energy systems. Monitoring their **State of Charge (SOC)** is essential for reliability, safety, and efficiency.  
Using NASA’s publicly available Li-ion battery degradation dataset, this project builds a complete pipeline to:

1. Clean and preprocess raw experimental data.  
2. Engineer relevant electrochemical features (e.g., impedance spectra, cycle counts).  
3. Train and evaluate machine learning and deep learning models to predict SOC.  

---

## Dataset

**Source:** NASA Ames Prognostics Data Repository — Li-ion Battery Aging Datasets.  
The dataset includes voltage, current, temperature, and impedance data across multiple charge/discharge cycles.

**Cleaning steps included:**
- Removed faulty or prematurely failed cells (e.g., Cells 29–32, 49–54).  
- Standardized frequency points for EIS data (0.1 Hz – 5 kHz, 48 points).  
- Merged multi-file cycle data into a unified format.  
- Handled missing data and outliers.

---

## Methods and Tools

- **Data Processing:** pandas, numpy, sqlite3  
- **Visualization:** matplotlib, plotly, seaborn  
- **Feature Engineering:** EIS feature extraction, normalization, polynomial features  
- **Modeling:** RandomForestRegressor, XGBRegressor, LGBMRegressor, LSTM  
- **Evaluation:** R², RMSE, cross-validation  
- **Environment:** Google Colab / Python 3.10  

---

## Modeling Workflow

1. **Preprocessing**  
   - Standardized frequency responses.  
   - Removed noisy measurements and end-of-life data.  
   - Scaled features using StandardScaler.

2. **Feature Engineering**  
   - Created polynomial and statistical features from impedance data.  
   - Applied PCA for dimensionality reduction when applicable.

3. **Modeling and Validation**  
   - Compared regression models (Random Forest, XGBoost, LightGBM).  
   - Experimented with LSTM neural networks for temporal SOC prediction.  
   - Evaluated performance using cross-validation and RMSE.

4. **Evaluation**  
   - Visualized predicted vs. true SOC values.  
   - Assessed feature importance for interpretability.

---

## Results Summary

- Produced a cleaned dataset suitable for downstream SOC and SOH modeling.  
- Gradient boosting and LSTM models achieved strong predictive performance.  
- Demonstrated interpretable correlations between impedance features and SOC.  

(*Exact metrics can be added once finalized from model outputs.*)

---

## Key Learnings

- Handling complex, multi-file scientific datasets.  
- Integrating multiple machine learning frameworks in one pipeline.  
- Building interpretable regression models for physical systems.  
- Applying time-series modeling (LSTM) to electrochemical data.

---

## Future Work

- Extend to **State of Health (SOH)** prediction using cycle degradation data.  
- Incorporate additional electrochemical features (e.g., Nyquist plot descriptors).  
- Optimize deep learning architectures for real-time embedded systems.

---

**Author:** *Prameswara Hanggoro*  
**Contact:** prameswarahanggoro@gmail.com*  
