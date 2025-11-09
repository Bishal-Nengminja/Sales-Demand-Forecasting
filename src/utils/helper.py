from sqlalchemy import create_engine
import pandas as pd

def make_conn_string(user, password, host, port, db):
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def read_sql_to_df(conn_string, query):
    engine = create_engine(conn_string)
    df = pd.read_sql_query(query, engine)
    return df
