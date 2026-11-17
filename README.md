# Data Center Power and Cooling Load Forecasting

A time-series forecasting notebook that predicts a data center's hourly **IT load**, **cooling power** and **PUE** (Power Usage Effectiveness) for the next 7 days, and compares classical models with a machine learning model:

- **Seasonal Naive** (baseline)
- **ARIMAX** (SARIMAX with weather as exogenous input)
- **Random Forest** (with lag features up to the daily and weekly cycle)

This was a solo course lab by Salma Ashraf. Each notebook cell is marked either `[LAB REQUIREMENT]` (part of the required lab structure) or `[STUDENT IMPROVEMENT]` (extra work beyond the lab).

## Data

The notebook builds its own dataset; no data files are included in this repository (see [`data/README.md`](data/README.md)).

- **Weather:** real hourly outdoor temperature and relative humidity for Cairo, July 2024 to June 2025, from the [Open-Meteo Historical Weather API](https://open-meteo.com/) (licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). If the API is unreachable, the notebook falls back to synthetic weather.
- **IT load:** **simulated**, not measured. It combines a daily cycle, a weekend dip, slow capacity growth, random AI-training bursts and noise.
- **Cooling power, total power and PUE:** derived from the simulated IT load and the real outdoor temperature using simple physics-inspired rules.

The results below therefore describe models trained on partly simulated data, not a real facility.

## Notebook structure

1. Setup
2. Data
3. Trend
4. ACF (auto-correlation)
5. Seasonality (STL decomposition, period = 24 h)
6. Stationarity (ADF + KPSS)
7. ACF / PACF after differencing
8. Train / test split (by time; the last 168 hours are the test week)
9. Models
10. Evaluation (one-step-ahead and 7-day-ahead)
11. Student extensions (residual checks, prediction intervals, PUE forecast, cooling regression, cooling-capacity scenario)
12. Conclusions

## Setup and how to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib statsmodels scikit-learn requests jupyter
jupyter notebook datacenter_forecasting.ipynb
```

Run all cells. Internet access is only needed to download the weather data. Running the notebook writes `datacenter_hourly_2024_2025.csv` to the working directory.

### About `build_notebook.py`

The notebook was generated from `build_notebook.py`, which assembles the cells with `nbformat`. You do not need it to run the notebook. To regenerate the notebook from the script, install `nbformat` and change the output path on the last line of the script (it currently points to `/home/user/DataCenter_Forecasting/`).

## Results

These numbers come from the saved outputs of the notebook run included in this repository (test week: 24–30 June 2025).

**7-day-ahead forecast**

| Target | Model | MAE | RMSE | MAPE % |
|---|---|---|---|---|
| Cooling power (kW) | ARIMAX | 23.39 | 35.24 | 6.51 |
| | Random Forest | 25.66 | 39.48 | 6.62 |
| | Seasonal Naive | 62.72 | 69.68 | 18.06 |
| IT load (kW) | Random Forest | 69.80 | 132.42 | 3.88 |
| | ARIMAX | 183.51 | 197.19 | 11.54 |
| | Seasonal Naive | 358.53 | 381.42 | 22.89 |

**One-step-ahead forecast:** Random Forest had the lowest RMSE for both cooling power (22.67 kW) and IT load (81.48 kW).

**Other outputs**

- Best-model residuals: Ljung-Box p-value 0.62 (24 lags), so no clear remaining pattern.
- 95% prediction interval coverage on the test week: 92.3%.
- PUE 7-day forecast: MAE 0.0110, MAPE 0.86%.
- Cooling-capacity scenario (limit = 95th percentile of training cooling power, 478.9 kW): 27 actual hours above the limit, 17 forecast by the best model, 74 in a +20% IT load scenario.

## Status

Completed course lab. Not actively maintained.

## License

[MIT](LICENSE) © 2026 Salma Ashraf

Weather data: © Open-Meteo, CC BY 4.0.
