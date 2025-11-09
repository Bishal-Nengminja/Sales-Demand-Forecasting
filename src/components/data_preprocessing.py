import pandas as pd
import re
from pathlib import Path
from sqlalchemy import create_engine
from src.utils.logger import get_logger
from src.config.db_config import DB_CONFIG, TABLE_NAME
import argparse

logger = get_logger(__name__)

def to_snake_case(s: str) -> str:
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_").lower()

def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Filling nulls: numeric->median, categorical->mode, datetime->ffill/bfill")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)
            logger.debug(f"Filled numeric {col} with median {med}")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            logger.debug(f"Filled datetime {col}")
        else:
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
                logger.debug(f"Filled categorical {col} with mode {mode.iloc[0]}")
            else:
                df[col] = df[col].fillna("")
                logger.debug(f"Filled categorical {col} with ''")
    return df

def preprocess_and_upload(input_path: str, output_path: str, upload_to_db: bool = True):
    logger.info(f"Starting preprocessing: {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Read {len(df)} rows from {input_path}")
    df.columns = [to_snake_case(c) for c in df.columns]
    logger.info(f"Columns after snake_case: {df.columns.tolist()}")
    # parse date-like cols heuristically
    for c in df.columns:
        if 'date' in c or 'time' in c:
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
                logger.info(f"Parsed {c} as datetime")
            except Exception:
                logger.warning(f"Could not parse {c} as datetime")
    df = fill_nulls(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote cleaned CSV to {output_path}")

    if upload_to_db:
        try:
            conn_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
            engine = create_engine(conn_str)
            df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
            logger.info(f"Uploaded cleaned data to PostgreSQL table `{TABLE_NAME}`")
        except Exception as e:
            logger.error("Failed to upload to PostgreSQL", exc_info=True)
            raise

    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no_upload", action="store_true", help="Don't upload to Postgres")
    args = p.parse_args()
    preprocess_and_upload(args.input, args.output, upload_to_db=not args.no_upload)
