import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
import argparse

logger = get_logger(__name__)

def make_dataset(input_csv: str, output_dir: str, test_ratio: float = 0.2):
    df = pd.read_csv(input_csv, parse_dates=True)
    date_col = next((c for c in df.columns if 'date' in c), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)
    else:
        df = df.sort_index()

    n = len(df)
    split = int(n * (1 - test_ratio))
    train = df.iloc[:split]
    test = df.iloc[split:]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(Path(output_dir)/"train.csv", index=False)
    test.to_csv(Path(output_dir)/"test.csv", index=False)
    logger.info(f"Created train ({len(train)}) and test ({len(test)}) in {output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--test_ratio", default=0.2, type=float)
    args = p.parse_args()
    make_dataset(args.input, args.output, args.test_ratio)
