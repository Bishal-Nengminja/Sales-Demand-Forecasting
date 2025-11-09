# Sales Demand Forecasting — Prophet + Logging + Dagshub

This repository contains a full data science pipeline:
- preprocess raw CSV (snake_case + fill nulls)
- automatically upload cleaned data to PostgreSQL
- feature engineering (optional)
- train Prophet forecast; log to MLflow (Dagshub)
- version data & model with DVC (remote: Dagshub)
- logging (console + file)

## Quick start (after files in place)
1. Create env:
- conda env create -f environment.yml
- conda activate sales_forecast_prophet
- pip install -r requirements.txt

2. Edit `.env` with your Dagshub values and `src/config/db_config.py` with DB credentials.
3. Place raw CSV at `data/raw/sales_original.csv`.
4. Start MLflow UI (if you want local fallback): `bash scripts/start_mlflow.sh`
5. Run pipeline:

python src/main.py

6. Use DVC to add artifacts and push to Dagshub.

See scripts and src/ for exact commands and file structure.
