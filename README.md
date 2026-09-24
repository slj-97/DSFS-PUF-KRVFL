# DSFS–PUF–KRVFL ablation

This folder contains the MATLAB code and prepared input files for the paper's four-model ablation on six time series. It compares KRVFL, DSFS–KRVFL, PUF–KRVFL, and DSFS–PUF–KRVFL.

## Run

Use MATLAB R2024a or later. No external MATLAB toolbox is required.

1. Open MATLAB and set the current folder to this directory.
2. Run:

   ```matlab
   run_ablation_6ds
   ```

The driver uses chronological 70/30 train/test splits, three expanding validation folds, and seeds 42–51. It reports RMSE, MAE, MAPE, R², and Willmott's index (WI).

Results are written to `results/ablation_6ds_confirm/` as per-dataset run and summary CSV files, a MAT file, and a run log. The `results/` directory is ignored by Git so generated outputs are not uploaded accidentally.

## Files

- `run_ablation_6ds.m`: experiment driver.
- `dpfk_dataset_config.m`: input columns, target columns, and leakage exclusions for these six datasets.
- `+dpfk/`: model, validation, feature-selection, and metric functions used by the driver.
- `data/`: prepared inputs used by the experiment.

## Data sources

The included files are prepared versions of these public datasets:

| File | Dataset/source |
|---|---|
| `PedalMeWeeklyDemand.csv` | [UCI Pedal Me Bicycle Deliveries](https://archive.ics.uci.edu/dataset/847/pedal+me+bicycle+deliveries); aggregated to weekly demand. |
| `Daily_Demand_Forecasting_Orders.csv` | [UCI Daily Demand Forecasting Orders](https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders). |
| `stock.csv` | [KEEL Stock regression dataset](https://sci2s.ugr.es/keel/dataset.php?cod=77). |
| `hour.xlsx` | [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset); hourly records. |
| `SeoulBike_2000.csv` | [UCI Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand); first 2,000 prepared rows. |
| `SteelEnergy_2000.csv` | [UCI Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption); first 2,000 prepared rows. |

No reuse license is included. Check the source dataset terms and choose a repository license before inviting reuse.
