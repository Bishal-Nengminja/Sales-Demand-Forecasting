import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
import argparse

logger = get_logger(__name__)

def add_calendar_features(df, date_col='sale_date'):
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    logger.info("Added calendar features")
    return df

def create_lag_features(df, group_cols=None, target_col='sales', lags=[1,7,28]):
    if group_cols is None:
        group_cols = []
    df = df.sort_values(group_cols + ['sale_date'] if group_cols else ['sale_date'])
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag) if group_cols else df[target_col].shift(lag)
    df = df.fillna(0)
    logger.info("Created lag features and filled missing")
    return df

def process(input_csv, output_csv):
    df = pd.read_csv(input_csv, parse_dates=['sale_date'])
    df = add_calendar_features(df, 'sale_date')
    group_cols = ['store_id','product_id'] if {'store_id','product_id'}.issubset(df.columns) else []
    df = create_lag_features(df, group_cols=group_cols, target_col='sales')
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote features to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    process(args.input, args.output)
