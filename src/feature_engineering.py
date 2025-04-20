import pandas as pd

def basic_features(transactions: pd.DataFrame, token_transfers: pd.DataFrame, dex_swaps: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts simple features for each address from the provided tables.
    Returns a DataFrame indexed by address.
    """
    # Example: count of transactions sent
    tx_counts = transactions.groupby('FROM_ADDRESS').size().rename('tx_count')
    # Example: count of unique counterparties
    unique_tos = transactions.groupby('FROM_ADDRESS')['TO_ADDRESS'].nunique().rename('unique_tos')
    # Example: total ETH sent
    total_eth_sent = transactions.groupby('FROM_ADDRESS')['VALUE'].sum().rename('total_eth_sent')
    # Example: token diversity
    token_diversity = token_transfers.groupby('FROM_ADDRESS')['CONTRACT_ADDRESS'].nunique().rename('token_diversity')
    # Example: DEX swap count
    swap_count = dex_swaps.groupby('ORIGIN_FROM_ADDRESS').size().rename('swap_count')

    # Merge features
    features = pd.concat([
        tx_counts, unique_tos, total_eth_sent, token_diversity, swap_count
    ], axis=1).fillna(0)
    features.index.name = 'ADDRESS'
    return features.reset_index()
