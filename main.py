import os
import sys
from src.data_loader import load_labeled_addresses, load_test_addresses, load_transactions, load_token_transfers, load_dex_swaps
from src.feature_engineering import basic_features
from src.model import train_model
from src.predict import make_predictions

# Allow the user to specify chain via command-line argument (default: base)
if len(sys.argv) > 1:
    chain = sys.argv[1].lower()
    if chain not in ["base", "ethereum"]:
        raise ValueError("Chain must be 'base' or 'ethereum'")
else:
    chain = "base"

DATA_DIR = os.path.join("data", chain)
LABELS_PATH = os.path.join(DATA_DIR, 'train_addresses.parquet')
TEST_PATH = os.path.join(DATA_DIR, 'test_addresses.parquet')
TX_PATH = os.path.join(DATA_DIR, 'transactions.parquet')
TT_PATH = os.path.join(DATA_DIR, 'token_transfers.parquet')
DEX_PATH = os.path.join(DATA_DIR, 'dex_swaps.parquet')

if __name__ == '__main__':
    print(f"Running pipeline for chain: {chain}")
    # Load data
    labels = load_labeled_addresses(LABELS_PATH)
    test_addrs = load_test_addresses(TEST_PATH)
    transactions = load_transactions(TX_PATH)
    token_transfers = load_token_transfers(TT_PATH)
    dex_swaps = load_dex_swaps(DEX_PATH)

    # Feature engineering
    features = basic_features(transactions, token_transfers, dex_swaps)
    features = features.merge(labels, left_on='ADDRESS', right_on='ADDRESS', how='left')
    train_df = features[features['LABEL'].notnull()]
    test_df = features[features['ADDRESS'].isin(test_addrs['ADDRESS'])]

    X_train = train_df.drop(['ADDRESS', 'LABEL'], axis=1)
    y_train = train_df['LABEL'].astype(int)  # Convert Decimal to int
    X_test = test_df.drop(['ADDRESS', 'LABEL'], axis=1, errors='ignore')
    addresses = test_df['ADDRESS']

    # Train model
    model, val_auc = train_model(X_train, y_train)
    print(f'Validation AUC: {val_auc:.4f}')

    # Predict and save
    out_file = f'submission_{chain}.csv' if chain != 'base' else 'submission.csv'
    make_predictions(model, X_test, addresses, out_path=out_file)
