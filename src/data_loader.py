import pandas as pd
from pathlib import Path

def load_labeled_addresses(path: str) -> pd.DataFrame:
    """Load labeled Sybil addresses from parquet."""
    return pd.read_parquet(path)

def load_test_addresses(path: str) -> pd.DataFrame:
    """Load test addresses from parquet."""
    return pd.read_parquet(path)

def load_transactions(path: str) -> pd.DataFrame:
    """Load transactions data from parquet."""
    return pd.read_parquet(path)

def load_token_transfers(path: str) -> pd.DataFrame:
    """Load token transfers data from parquet."""
    return pd.read_parquet(path)

def load_dex_swaps(path: str) -> pd.DataFrame:
    """Load DEX swaps data from parquet."""
    return pd.read_parquet(path)
