import joblib, pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict(model_path: str, input_csv: str, output_csv: str):
    logger.info(f"Loading model {model_path}")
    m = joblib.load(model_path)
    df = pd.read_csv(input_csv, parse_dates=True)
    date_col = next((c for c in df.columns if 'date' in c), None)
    if date_col is None:
        raise ValueError("No date column found")
    future = pd.DataFrame({'ds': pd.to_datetime(df[date_col])})
    forecast = m.predict(future)
    df['yhat'] = forecast['yhat'].values
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote predictions to {output_csv} ({len(df)} rows)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    predict(args.model, args.input, args.output)
