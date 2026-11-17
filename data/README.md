# Data

No data files are stored in this repository. The notebook creates the dataset itself when you run it.

## `datacenter_hourly_2024_2025.csv`

Running `datacenter_forecasting.ipynb` writes this file to the working directory. It has 8,760 hourly rows (2024-07-01 00:00 to 2025-06-30 23:00) and these columns:

| Column | Meaning | Source |
|---|---|---|
| `temp_c` | Outdoor temperature (°C), Cairo | Open-Meteo Historical Weather API (real data) |
| `rh` | Outdoor relative humidity (%), Cairo | Open-Meteo Historical Weather API (real data) |
| `temp_ema3` | 3-hour exponential moving average of `temp_c` | Derived |
| `it_kw` | IT load (kW) | Simulated |
| `cooling_kw` | Cooling power (kW) | Derived from `it_kw` and `temp_ema3` |
| `total_kw` | IT + cooling + distribution overhead (kW) | Derived |
| `pue` | Power Usage Effectiveness = `total_kw / it_kw` | Derived |

If the Open-Meteo API cannot be reached, the notebook uses synthetic weather instead and prints `Weather source: synthetic fallback`.

## Attribution

Weather data by [Open-Meteo](https://open-meteo.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
