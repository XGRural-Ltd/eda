import pandas as pd
import numpy as np
import json

def to_df(obj):
    """Convert stored object (str/json/dict/DataFrame) -> DataFrame or None."""
    if obj is None:
        return None
    try:
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, str):
            # assume orient='split' JSON
            return pd.read_json(obj, orient='split')
        if isinstance(obj, dict) and {'data','columns'}.issubset(set(obj.keys())):
            return pd.DataFrame(**obj)
        return pd.DataFrame(obj)
    except Exception:
        return None

def df_to_store(df):
    """Serialize DataFrame to dict with orient='split' (safe for dcc.Store)."""
    if df is None:
        return None
    return df.to_dict(orient='split')

def ensure_numeric_df(df):
    if df is None:
        return None
    return df.select_dtypes(include=np.number).copy()