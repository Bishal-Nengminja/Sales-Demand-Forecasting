from src.utils.logger import get_logger
from src.utils.helper import read_sql_to_df, make_conn_string
from src.config.db_config import DB_CONFIG
from pathlib import Path
import argparse

logger = get_logger(__name__)

def ingest_from_db(query: str, out_csv: str):
    conn_str = make_conn_string(DB_CONFIG['user'], DB_CONFIG['password'],
                                DB_CONFIG['host'], DB_CONFIG['port'], DB_CONFIG['dbname'])
    logger.info("Ingesting data from Postgres")
    df = read_sql_to_df(conn_str, query)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_csv}")
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="SELECT * FROM sales_data")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ingest_from_db(args.query, args.out)
