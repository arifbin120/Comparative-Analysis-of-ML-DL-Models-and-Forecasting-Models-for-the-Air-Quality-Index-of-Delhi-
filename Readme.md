================================================================================
  COMPARATIVE ANALYSIS OF ML, DEEP LEARNING, AND FORECASTING MODELS FOR AQI
  Delhi Air Quality Index Prediction (2015-2024)
================================================================================

AUTHORS
-------
Sumit Kumar & Arif Bin Mushtaq
Department of Statistics, Faculty of Mathematical Sciences
University of Delhi, Delhi
Emails: sumitkumar@stats.du.ac.in | arifbinmushtaqq@stats.du.ac.in


--------------------------------------------------------------------------------
PROJECT OVERVIEW
--------------------------------------------------------------------------------

This project presents a comprehensive comparative analysis of Machine Learning
(ML), Deep Learning (DL), and Time Series Forecasting methods for predicting
the Air Quality Index (AQI) of Delhi, India -- one of the most polluted cities
in the world.

The study uses 10 years of daily AQI data (2015-2024) sourced from the Central
Pollution Control Board (CPCB), India, and addresses three core problems:

  1. How did AQI trends change during the 2020 COVID-19 lockdown compared to
     previous years (2015-2019) and later years (2021-2024)?

  2. Which ML/DL model best predicts Delhi's AQI given historical pollutant
     concentration data?

  3. Can time series forecasting methods reliably capture the seasonal and
     cyclical pollution trends in Delhi's AQI over a 10-year period?


--------------------------------------------------------------------------------
DATASET
--------------------------------------------------------------------------------

Source   : Central Pollution Control Board (CPCB), India (via Kaggle)
Period   : January 2015 - December 2024
Records  : 3,653 daily observations
Features : PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene,
           Xylene, Temperature, Humidity, Wind Speed, AQI, AQI Bucket, Datetime

Dataset Links:
  1. https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard/caaqm-landing/caaqm-data-availability
  2. https://airquality.cpcb.gov.in/AQI_India/
  3. https://www.aqi.in/dashboard/india/delhi/new-delhi  (live AQI)
  4. https://drive.google.com/drive/u/0/folders/1qWhVRV45fq5ZZ2HICBZ2b-vtnveHBmdP

AQI Categories (CPCB / NAAQS):
  0-50     Good          Minimal health impact
  51-100   Satisfactory  Minor breathing discomfort for sensitive people
  101-200  Moderate      Breathing discomfort for lung/asthma/heart patients
  201-300  Poor          Breathing discomfort for most on prolonged exposure
  301-400  Very Poor     Respiratory illness on prolonged exposure
  401-500  Severe        Affects healthy people; seriously impacts those with
                         existing diseases


--------------------------------------------------------------------------------
KEY FINDINGS
--------------------------------------------------------------------------------

>> EDA Highlights
  - Mean AQI   : 208.9 (Very Poor category)
  - Median AQI : 177.5 (Poor category)
  - PM10 (219.4 ug/m3) and PM2.5 (149.8 ug/m3) are the dominant pollutants,
    far exceeding WHO guidelines
  - Strong seasonal pattern: peaks in Nov-Jan (winter), drops in Jun-Sep (monsoon)
  - Worst monthly average: November (AQI 407.7, Severe)
  - Bimodal distribution: AQI ~100 in summer/monsoon, ~320 in winter
  - AQI category breakdown: Moderate 35.4%, Very Poor 21.4%, Poor 18.5%,
    Satisfactory 16.5%, Severe 5.5%, Good 2.8%

>> COVID-19 Lockdown Effect (April-June 2020)
  - Lockdown Avg AQI  : 55.0
  - 10-Year Baseline  : 126.8 (same calendar period)
  - Improvement       : 56.6% reduction (71.8 AQI units)
  - Annual mean AQI for 2020: ~175 -- the lowest in the entire decade
  - Effect was temporary; AQI rebounded to historical norms once restrictions lifted
<img width="1855" height="1499" alt="image" src="https://github.com/user-attachments/assets/4643afa9-c3c1-4d8f-a0e0-08a6a2f4eef8" />

>> Top Pollutant Correlations with AQI
  PM2.5        r =  0.9763   Primary AQI driver (particulate matter)
  PM10         r =  0.9744   Primary AQI driver (particulate matter)
  CO           r =  0.9549   Combustion indicator
  NOx          r =  0.9547   Vehicular emissions
  NO           r =  0.9369   Combustion by-product
  Temperature  r = -0.73     Negative -- cleaner air in warm months
  Wind Speed   r = -0.72     Negative -- disperses pollutants


--------------------------------------------------------------------------------
MODELS AND RESULTS
--------------------------------------------------------------------------------

== REGRESSION (Continuous AQI Value Prediction) ==

  Model               R2      Adj R2   MAE     RMSE    CV R2
  ----------------------------------------------------------------
  Ridge Regression   0.9859  0.9856   10.86   13.51   0.9868  << BEST
  XGBoost            0.9847  0.9843   11.25   14.08   0.9850
  Gradient Boosting  0.9844  0.9841   11.40   14.20   0.9845
  Random Forest      0.9833  0.9830   11.61   14.68   0.9836

  Winner: Ridge Regression
  Reason: AQI is largely a linear combination of standardised pollutant
  concentrations, directly mirroring the CPCB formula structure. All 12
  features are useful (gap between R2 and Adj R2 < 0.003). No overfitting
  (CV R2 tracks test R2 within 0.001-0.002).


== CLASSIFICATION (AQI Bucket / Category Prediction) ==

  Model                Accuracy  Macro F1  Wtd F1   CV Acc
  ----------------------------------------------------------
  XGBoost             0.8892    0.8654    0.8892   0.8566  << BEST
  Gradient Boosting   0.8851    0.8537    0.8851   0.8525
  Logistic Regression 0.8728    0.8418    0.8728   0.8569
  Random Forest       0.8632    0.8121    0.8632   0.8481

  Winner: XGBoost -- leads across all four metrics.
  Note: Logistic Regression outperforms Random Forest despite being simpler,
  consistent with the near-linear structure seen in regression results.

  Per-class F1 Highlights (XGBoost):
    Very Poor : 0.919
    Severe    : 0.923  (best)
    Good      : 0.737  (hardest -- fewest training samples)


== DEEP LEARNING -- ARTIFICIAL NEURAL NETWORK (Classification) ==

  Architecture used: MLP Classifier (scikit-learn)

  Model             Val Acc  Test Acc  Macro F1  Wtd F1   CV Acc  Epochs
  -----------------------------------------------------------------------
  ANN -- Deep       0.8446   0.8923    0.8640    0.8924   0.8616   65  << BEST
  ANN -- Regularized 0.8208  0.8741    0.8441    0.8740   0.8499   68
  ANN -- Shallow    0.8245   0.8686    0.7972    0.8664   0.8558   60

  Best Architecture: Input(12) -> 256 -> 128 -> 64 -> 32 -> Output(6)
  Activation: ReLU | Optimizer: Adam | L2 regularization alpha = 0.0005

  Dataset Split for ANN:
    Training   : 2,558 samples (70%)
    Validation :   547 samples (15%)
    Test       :   548 samples (15%)

  ANN Deep -- Regression (continuous AQI value):
    R2 = 0.9862 | MAE = 10.68 | RMSE = 13.34
    Interpretation: 98.62% of AQI variance explained; avg error ~10-11 units
    on a 0-500 scale (relative error ~2.1%).

  Cross-Model Confusion Matrix Summary (Recall by AQI Category):
    AQI Category   Shallow  Deep   Regularized  Key Observation
    Good           0.27     0.60   0.80         Biggest gap between models
    Moderate       0.88     0.90   0.89         High support, all models strong
    Poor           0.95     0.94   0.98         Consistently excellent
    Satisfactory   0.81     0.84   0.78         Boundary class, moderate difficulty
    Severe         0.90     0.93   0.70         Regularized trades Severe for Good
    Very Poor      0.89     0.91   0.89         Consistent across all models


== TIME SERIES FORECASTING (2025-2027 Projection) ==

  Method                   MAE     RMSE    MAPE(%)  R2       DA(%)
  ----------------------------------------------------------------
  Holt-Winters (TES)       6.95    7.74    4.51    0.9941   90.91  << BEST RMSE
  ML Regression Poly+FFT  12.32   14.30    6.20    0.9800  100.00  << BEST DA
  SES (alpha=0.90)         56.63   72.41   27.66   0.4879   81.82
  ARIMA (3,1,2)           108.70  125.58   92.03  -0.5403   54.55
  SMA (k=6)               110.96  130.76   66.53  -0.6699   45.45

  Holt-Winters Parameters: alpha=0.5 (level), beta=0.1 (trend), gamma=0.4 (seasonality)
  Seasonal period: m = 365 days

  Key Insight: Models that incorporate seasonality (TES, ML Regression) dramatically
  outperform those that do not (SMA, SES, ARIMA). ARIMA captures trend but
  significantly understates seasonal fluctuations.

  3-Year Ensemble Forecast:
    Year   Avg AQI   Peak AQI   Low AQI
    2025    219.9     349.8      142.3
    2026    225.3     351.0      148.4
    2027    229.7     352.8      154.5

  Trend: Gradual deterioration across all levels. Even the cleanest months
  are becoming more polluted (minimum AQI rising from 142.3 to 154.5).


--------------------------------------------------------------------------------
MODEL SELECTION GUIDE
--------------------------------------------------------------------------------

  Use Case                        Recommended Model          Reason
  ----------------------------------------------------------------
  Best overall AQI prediction     ANN -- Deep                Leads all 5 metrics
  Detecting 'Good' air quality    ANN -- Regularized         Recall 0.80 for Good class
  Lightweight / fast deployment   ANN -- Shallow             Fewest parameters
  Exact AQI number (regression)   Ridge Regression           Highest R2, lowest error
  AQI category classification     XGBoost                    Best accuracy and F1
  Seasonal AQI forecasting        Holt-Winters (TES)         Best RMSE and R2
  Trend + direction forecasting   ML Regression Poly+Fourier Perfect DA (100%)


--------------------------------------------------------------------------------
AQI FORMULA (Indian NAAQS Framework)
--------------------------------------------------------------------------------

Step 1 -- Sub-index for each pollutant p (piecewise linear interpolation):

  Ip = ((IHI - ILO) / (BPHI - BPLO)) * (Cp - BPLO) + ILO

  Where:
    Cp   = Observed concentration of pollutant p
    BPHI = Upper breakpoint concentration >= Cp
    BPLO = Lower breakpoint concentration <= Cp
    IHI  = AQI value corresponding to BPHI
    ILO  = AQI value corresponding to BPLO

Step 2 -- Final AQI:
  AQI = max(I1, I2, I3, ..., In)

  Rules:
    - PM2.5 and PM10 are mandatory pollutants
    - Minimum 3 pollutants required to compute AQI
    - At least 16 hours of data required per sub-index
    - Up to 8 pollutants considered under NAAQS


--------------------------------------------------------------------------------
TECH STACK
--------------------------------------------------------------------------------

Language  : Python 3
Platform  : Google Colab

Core Libraries:
  Data        : pandas, numpy, scipy
  ML Models   : scikit-learn (Ridge, RandomForest, GradientBoosting, MLP,
                Pipeline, StandardScaler, LabelEncoder)
  Boosting    : xgboost
  Forecasting : Custom OLS-based ARIMA, scikit-learn (PolynomialFeatures,
                Ridge), numpy (Fourier features)
  Visualization: matplotlib, seaborn

Colab Notebook:
  https://colab.research.google.com/drive/1w1vbbFcPLc3l1VEwFxl5Q1D2jaEl5H9k?usp=drive_link

Additional Resources:
  https://scikit-learn.org/stable/
  https://scikit-learn.org/stable/modules/neural_networks_supervised.html
  https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html


--------------------------------------------------------------------------------
SUGGESTED PROJECT STRUCTURE
--------------------------------------------------------------------------------

  /
  |-- README.txt
  |-- data/
  |   |-- UPDATED_DELHI_AQI_DATA.csv
  |-- notebooks/
  |   |-- 01_EDA.ipynb
  |   |-- 02_ML_Models.ipynb
  |   |-- 03_ANN_Deep_Learning.ipynb
  |   |-- 04_Forecasting.ipynb
  |-- src/
  |   |-- preprocessing.py
  |   |-- ml_models.py
  |   |-- ann_models.py
  |   |-- forecasting.py
  |-- results/
  |   |-- figures/
  |-- requirements.txt


--------------------------------------------------------------------------------
REQUIREMENTS (requirements.txt)
--------------------------------------------------------------------------------

  pandas
  numpy
  scipy
  scikit-learn
  xgboost
  matplotlib
  seaborn


--------------------------------------------------------------------------------
SELECTED REFERENCES
--------------------------------------------------------------------------------

[1]  Liao et al. (2020). Deep Learning for Air Quality Forecasts: a Review.
     Current Pollution Reports, 6, 399-409.
     https://doi.org/10.1007/s40726-020-00159-z

[2]  Abu Sadat et al. (2024). Forecasting AQI in Sylhet: ML and DL Models.
     https://www.researchgate.net/publication/387090243

[3]  Tirink (2025). XGBoost for daily AQI prediction, Turkiye.

[4]  WHO (2014). 7 million premature deaths annually linked to air pollution.

[5]  Montgomery et al. (2015). Introduction to Time Series Analysis. Wiley.

[6]  Winters, P.R. (1960). Forecasting sales by exponentially weighted
     moving averages. Management Science.

[7]  LeCun, Bengio, Hinton (2015). Deep learning. Nature, 521, 436-444.
     https://doi.org/10.1038/nature14539

Full reference list available in the paper.


--------------------------------------------------------------------------------
LICENSE
--------------------------------------------------------------------------------

This project is for academic and research purposes.
Data sourced from CPCB, India.
Please cite the authors if you use this work.

================================================================================
