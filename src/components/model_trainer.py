import os
import sys
import argparse
import pandas as pd
from prophet import Prophet
import joblib
import mlflow
import yaml
from dotenv import load_dotenv
from src.utils.logger import get_logger

# ✅ Initialize logger
logger = get_logger(__name__)

def load_config(params_path):
    """Load hyperparameters or configuration from YAML."""
    with open(params_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(train_csv, test_csv, params_path, mlflow_uri=None):
    """Train Prophet model and log results to MLflow."""
    # Load environment variables automatically
    load_dotenv()

    # Load training params
    cfg = load_config(params_path)
    logger.info("Loaded params from %s", params_path)

    # Connect to MLflow (with DagsHub credentials)
    try:
        tracking_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(cfg.get('experiment_name', 'sales_forecast_prophet_experiment'))
        logger.info(f"Using MLflow tracking URI: {tracking_uri}")

        # ✅ Check connection (try to start a dummy run)
        with mlflow.start_run(run_name="connection_test"):
            mlflow.log_param("connection_check", "ok")
        logger.info("✅ Connected to MLflow successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Could not connect to remote MLflow ({e}). Falling back to local tracking.")
        mlflow.set_tracking_uri("file:./mlflow_tracking")
        mlflow.set_experiment("sales_forecast_prophet_local")

    # Read data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    logger.info("Training data loaded: %s rows", len(train_df))
    logger.info("Testing data loaded: %s rows", len(test_df))

    # Prophet expects columns named 'ds' and 'y'
    train_df = train_df.rename(columns={'date': 'ds', 'sales': 'y'})
    test_df = test_df.rename(columns={'date': 'ds', 'sales': 'y'})

    # Train Prophet model
    m = Prophet(
        seasonality_mode=cfg.get('seasonality_mode', 'additive'),
        yearly_seasonality=cfg.get('yearly_seasonality', True),
        weekly_seasonality=cfg.get('weekly_seasonality', True),
        daily_seasonality=cfg.get('daily_seasonality', False),
        changepoint_prior_scale=cfg.get('changepoint_prior_scale', 0.05)
    )

    with mlflow.start_run(run_name="prophet_training"):
        m.fit(train_df)
        logger.info("✅ Prophet model training completed.")

        # Forecast on test data
        forecast = m.predict(test_df[['ds']])
        forecast['actual'] = test_df['y'].values

        # Calculate simple metrics
        forecast['error'] = forecast['yhat'] - forecast['actual']
        mae = abs(forecast['error']).mean()
        rmse = (forecast['error']**2).mean()**0.5

        logger.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Log to MLflow
        mlflow.log_params(cfg)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/prophet_model.pkl"
        joblib.dump(m, model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")

    logger.info("🎯 Training and MLflow logging complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Prophet Model for Sales Forecasting")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV")
    parser.add_argument("--params", required=True, help="Path to YAML params file")
    parser.add_argument("--mlflow_uri", required=False, help="Remote MLflow tracking URI")
    args = parser.parse_args()

    train(args.train_csv, args.test_csv, args.params, args.mlflow_uri)
