# 🌫️ Delhi AQI Prediction: ML, Deep Learning & Forecasting (2015–2024)

> A comprehensive comparative analysis of Machine Learning, Deep Learning, and Time Series Forecasting methods for predicting Delhi's Air Quality Index.

**Authors:** Sumit Kumar & Arif Bin Mushtaq  
**Institution:** Department of Statistics, Faculty of Mathematical Sciences, University of Delhi  
**Contact:** sumitkumar@stats.du.ac.in | arifbinmushtaqq@stats.du.ac.in  
**Data Source:** Central Pollution Control Board (CPCB), India  
**Period:** 2015–2024 (10 years of daily AQI data)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w1vbbFcPLc3l1VEwFxl5Q1D2jaEl5H9k?usp=drive_link)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Research Questions](#-research-questions)
- [Key Findings](#-key-findings)
- [Dataset & Features](#-dataset--features)
- [Models & Results](#-models--results)
  - [Regression](#1-regression--continuous-aqi-prediction)
  - [Classification](#2-classification--aqi-category-prediction)
  - [Deep Learning (ANN)](#3-deep-learning--ann)
  - [Time Series Forecasting](#4-time-series-forecasting)
- [Model Selection Guide](#-model-selection-guide)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [References](#-references)
- [License](#-license)

---

## 🔍 Project Overview

Delhi is one of the most polluted cities in the world. This project systematically benchmarks classical ML models, gradient boosting methods, Artificial Neural Networks (ANNs), and time series forecasting techniques to determine which approach best predicts Delhi's AQI — both as a continuous value and as a categorical label.

Three modelling paradigms are covered:

| Paradigm | Task | Best Model |
|---|---|---|
| ML Regression | Predict exact AQI value | Ridge Regression |
| ML / DL Classification | Predict AQI category (Good → Severe) | XGBoost / ANN-Deep |
| Time Series Forecasting | Project AQI trends (2025–2027) | Holt-Winters (TES) |

---

## ❓ Research Questions

1. **COVID-19 Effect** — How did AQI change during the 2020 lockdown compared to 2015–2019 baselines and post-lockdown years (2021–2024)?
2. **Best Predictive Model** — Which ML/DL model most accurately predicts Delhi's AQI from historical pollutant data?
3. **Forecasting Reliability** — Can time series methods reliably capture Delhi's seasonal and cyclical pollution patterns over a 10-year horizon?

---

## 📊 Key Findings

### Exploratory Data Analysis

| Metric | Value |
|---|---|
| Mean AQI | 208.9 (Very Poor) |
| Median AQI | 177.5 (Poor) |
| Worst Month | November — avg AQI **407.7** (Severe) |
| Best Season | Jun–Sep (Monsoon) |

**AQI Category Distribution**

| Category | Share |
|---|---|
| Moderate | 35.4% |
| Very Poor | 21.4% |
| Poor | 18.5% |
| Satisfactory | 16.5% |
| Severe | 5.5% |
| Good | 2.8% |

**Dominant Pollutants:** PM10 (219.4 µg/m³) and PM2.5 (149.8 µg/m³) — both far exceed WHO guidelines.  
**Seasonal Pattern:** Bimodal — AQI ~100 in summer/monsoon, ~320 in winter.

---

### COVID-19 Lockdown Effect (April–June 2020)

| Period | Avg AQI |
|---|---|
| Lockdown (Apr–Jun 2020) | **55.0** |
| 10-Year Baseline (same calendar period) | 126.8 |
| Improvement | **56.6% reduction (−71.8 AQI units)** |

Annual mean AQI for 2020 was ~175 — the **lowest in the entire decade**. The effect was temporary; AQI rebounded to historical norms once restrictions were lifted.

---

### Pollutant Correlations with AQI

| Pollutant | Pearson r | Role |
|---|---|---|
| PM2.5 | +0.9763 | Primary AQI driver |
| PM10 | +0.9744 | Primary AQI driver |
| CO | +0.9549 | Combustion indicator |
| NOx | +0.9547 | Vehicular emissions |
| NO | +0.9369 | Combustion by-product |
| Temperature | −0.73 | Cleaner air in warm months |
| Wind Speed | −0.72 | Disperses pollutants |

> AQI is primarily driven by **PM2.5, PM10, NO, and SO2** due to their higher correlation with the index.

---

## 🗂️ Dataset & Features

**File:** `data/UPDATED_DELHI_AQI_DATA.csv`

```
City | Datetime | Month | PM2.5 | PM10 | NO | NO2 | NOx | NH3 | CO
... | Xylene | Temperature_C | Humidity_pct | WindSpeed_kmh
AQI | AQI_Bucket | Year | DayOfWeek | Quarter | Season
```

---

## 🤖 Models & Results

### 1. Regression — Continuous AQI Prediction

| Model | R² | Adj R² | MAE | RMSE | CV R² |
|---|---|---|---|---|---|
| **Ridge Regression** ⭐ | **0.9859** | **0.9856** | **10.86** | **13.51** | **0.9868** |
| XGBoost | 0.9847 | 0.9843 | 11.25 | 14.08 | 0.9850 |
| Gradient Boosting | 0.9844 | 0.9841 | 11.40 | 14.20 | 0.9845 |
| Random Forest | 0.9833 | 0.9830 | 11.61 | 14.68 | 0.9836 |

---

### 2. Classification — AQI Category Prediction

Categories: **Good → Satisfactory → Moderate → Poor → Very Poor → Severe**

| Model | Accuracy | Macro F1 | Weighted F1 | CV Acc |
|---|---|---|---|---|
| **XGBoost** ⭐ | **0.8892** | **0.8654** | **0.8892** | **0.8566** |
| Gradient Boosting | 0.8851 | 0.8537 | 0.8851 | 0.8525 |
| Logistic Regression | 0.8728 | 0.8418 | 0.8728 | 0.8569 |
| Random Forest | 0.8632 | 0.8121 | 0.8632 | 0.8481 |

---

### 3. Deep Learning — ANN (MLP Classifier)

**Best Architecture:** `Input(12) → 256 → 128 → 64 → 32 → Output(6)`  
**Activation:** ReLU | **Optimizer:** Adam | **L2 alpha:** 0.0005

**Dataset Split:** Train 70% (2,558) | Val 15% (547) | Test 15% (548)

| Model | Val Acc | Test Acc | Macro F1 | Wtd F1 | CV Acc | Epochs |
|---|---|---|---|---|---|---|
| **ANN-Deep** ⭐ | **0.8446** | **0.8923** | **0.8640** | **0.8924** | **0.8616** | 65 |
| ANN-Regularized | 0.8208 | 0.8741 | 0.8441 | 0.8740 | 0.8499 | 68 |
| ANN-Shallow | 0.8245 | 0.8686 | 0.7972 | 0.8664 | 0.8558 | 60 |

**ANN Regression (continuous AQI):** R² = 0.9862 | MAE = 10.68 | RMSE = 13.34  
*98.62% of AQI variance explained; avg error ~10–11 units on a 0–500 scale (~2.1% relative error)*

**Recall by AQI Category (Confusion Matrix Summary):**

| AQI Category | Shallow | Deep | Regularized | Observation |
|---|---|---|---|---|
| Good | 0.27 | 0.60 | **0.80** | Biggest gap between models |
| Moderate | 0.88 | **0.90** | 0.89 | High support; all models strong |
| Poor | 0.95 | 0.94 | **0.98** | Consistently excellent |
| Satisfactory | 0.81 | **0.84** | 0.78 | Boundary class, moderate difficulty |
| Severe | 0.90 | **0.93** | 0.70 | Regularized trades Severe recall for Good |
| Very Poor | 0.89 | **0.91** | 0.89 | Consistent across all models |

---

### 4. Time Series Forecasting

**Horizon:** 2025–2027 | **Granularity:** Daily, then aggregated

| Method | MAE | RMSE | MAPE (%) | R² | DA (%) |
|---|---|---|---|---|---|
| **Holt-Winters (TES)** ⭐ | **6.95** | **7.74** | **4.51** | **0.9941** | 90.91 |
| ML Regression (Poly+FFT) | 12.32 | 14.30 | 6.20 | 0.9800 | **100.00** ⭐ |
| SES (α=0.90) | 56.63 | 72.41 | 27.66 | 0.4879 | 81.82 |
| ARIMA (3,1,2) | 108.70 | 125.58 | 92.03 | −0.5403 | 54.55 |
| SMA (k=6) | 110.96 | 130.76 | 66.53 | −0.6699 | 45.45 |

**Holt-Winters Parameters:** α=0.5 (level) | β=0.1 (trend) | γ=0.4 (seasonality) | m=365 days

> **Key Insight:** Models incorporating seasonality (TES, ML Regression) dramatically outperform those that do not. ARIMA captures trend but significantly understates seasonal fluctuations.

**3-Year Ensemble Forecast:**

| Year | Avg AQI | Peak AQI | Low AQI |
|---|---|---|---|
| 2025 | 219.9 | 349.8 | 142.3 |
| 2026 | 225.3 | 351.0 | 148.4 |
| 2027 | 229.7 | 352.8 | 154.5 |

⚠️ **Trend: Gradual deterioration.** Even the cleanest months are becoming more polluted — minimum AQI rising from 142.3 (2025) to 154.5 (2027).

---

## 🧭 Model Selection Guide

| Use Case | Recommended Model | Reason |
|---|---|---|
| Best overall AQI prediction | **ANN-Deep** | Leads all 5 metrics |
| Detecting 'Good' air quality | **ANN-Regularized** | Recall 0.80 for Good class |
| Lightweight / fast deployment | **ANN-Shallow** | Fewest parameters |
| Exact AQI number (regression) | **Ridge Regression** | Highest R², lowest error |
| AQI category classification | **XGBoost** | Best accuracy and F1 |
| Seasonal AQI forecasting | **Holt-Winters (TES)** | Best RMSE and R² |
| Trend + direction forecasting | **ML Regression (Poly+Fourier)** | Perfect DA (100%) |

---

## 📁 Project Structure

```
/
├── README.md
├── requirements.txt
├── data/
│   └── UPDATED_DELHI_AQI_DATA.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_ML_Models.ipynb
│   ├── 03_ANN_Deep_Learning.ipynb
│   └── 04_Forecasting.ipynb
├── src/
│   ├── preprocessing.py
│   ├── ml_models.py
│   ├── ann_models.py
│   └── forecasting.py
└── results/
    └── figures/
```

---

## 🛠️ Tech Stack

**Language:** Python 3 | **Platform:** Google Colab

```
pandas        # Data manipulation
numpy         # Numerical computing
scipy         # Scientific computing
scikit-learn  # ML models, preprocessing, evaluation
xgboost       # Gradient boosting
matplotlib    # Visualization
seaborn       # Statistical plots
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 📚 References

1. Liao et al. (2020). *Deep Learning for Air Quality Forecasts: a Review.* Current Pollution Reports, 6, 399–409. https://doi.org/10.1007/s40726-020-00159-z
2. Abu Sadat et al. (2024). *Forecasting AQI in Sylhet: ML and DL Models.* ResearchGate. https://www.researchgate.net/publication/387090243
3. Tirink (2025). *XGBoost for daily AQI prediction, Turkiye.*
4. WHO (2014). *7 million premature deaths annually linked to air pollution.*
5. Montgomery et al. (2015). *Introduction to Time Series Analysis.* Wiley.
6. Winters, P.R. (1960). *Forecasting sales by exponentially weighted moving averages.* Management Science.
7. LeCun, Bengio, Hinton (2015). *Deep learning.* Nature, 521, 436–444. https://doi.org/10.1038/nature14539

---

## 📄 License

This project is for **academic and research purposes only**.  
Data sourced from the Central Pollution Control Board (CPCB), India.  
Please **cite the authors** if you use this work.

---

<div align="center">
  <sub>Made with 🔬 by Sumit Kumar & Arif Bin Mushtaq · University of Delhi</sub>
</div>
