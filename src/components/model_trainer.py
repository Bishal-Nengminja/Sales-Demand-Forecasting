import yaml, joblib, os
from pathlib import Path
import mlflow
from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_prophet_df(df, date_col=None, target_col='sales'):
    if date_col is None:
        date_col = next((c for c in df.columns if 'date' in c), None)
    df = df.copy()
    df['ds'] = pd.to_datetime(df[date_col])
    df['y'] = df[target_col].astype(float)
    return df[['ds','y']]

def train(train_csv: str, test_csv: str, params_path: str, mlflow_uri: str = None):
    logger.info("Loading params")
    cfg = yaml.safe_load(open(params_path))
    model_cfg = cfg.get('model', {}).get('prophet', {})
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg.get('experiment_name', 'sales_prophet'))

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_prophet = prepare_prophet_df(train_df)
    test_prophet = prepare_prophet_df(test_df)

    m = Prophet(
        weekly_seasonality=model_cfg.get('weekly_seasonality', True),
        yearly_seasonality=model_cfg.get('yearly_seasonality', True),
        daily_seasonality=model_cfg.get('daily_seasonality', False),
        changepoint_prior_scale=model_cfg.get('changepoint_prior_scale', 0.05)
    )

    logger.info("Fitting Prophet model")
    m.fit(train_prophet)
    future = m.make_future_dataframe(periods=len(test_prophet), freq='D')
    forecast = m.predict(future)
    fc = forecast[['ds','yhat']].set_index('ds')
    merged = test_prophet.set_index('ds').join(fc, how='left').reset_index()
    rmse = sqrt(mean_squared_error(merged['y'], merged['yhat']))

    logger.info(f"Validation RMSE: {rmse:.4f}")
    # MLflow logging
    with mlflow.start_run():
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_params(model_cfg)
        Path('models').mkdir(parents=True, exist_ok=True)
        model_path = Path('models') / 'prophet_model.pkl'
        joblib.dump(m, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info(f"Model saved to {model_path} and logged to MLflow")

    return str(model_path), float(rmse)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--params", default="params.yaml")
    p.add_argument("--mlflow_uri", default=None)
    args = p.parse_args()
    train(args.train_csv, args.test_csv, args.params, args.mlflow_uri)
