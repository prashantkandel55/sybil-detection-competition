# Sybil Detection Competition: Baseline Pipeline

This repository provides a baseline machine learning pipeline for the Sybil Detection competition (April 2025) on both Base and Ethereum datasets.

## Project Structure
```
sybil/
├── data/
│   ├── base/         # Base chain data (parquet files)
│   └── ethereum/     # Ethereum chain data (parquet files)
├── src/              # Source code
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── predict.py
├── main.py           # Pipeline entry point
├── requirements.txt  # Python dependencies
├── submission.csv                # Output for Base
├── submission_ethereum.csv       # Output for Ethereum
└── README.md
```

## Data Table Relationships

- **Transactions Table:**
  - Contains blockchain transaction records: `Tx_Hash`, `From_Address`, `To_Address`, `Value`, etc.
- **Token Transfers Table:**
  - Contains ERC-20 token transfer records: `Tx_Hash`, `Origin_From_Address`, `Origin_To_Address`, `From_Address`, `To_Address`, `Contract_Address`, `Amount_Precise`, `Amount_USD`, etc.
- **Relationship:**
  - The tables are linked by `Tx_Hash`.
  - Each transaction in the Transactions table can have one or more associated token transfers in the Token Transfers table.
  - This allows you to join or merge the tables to get a full picture of what happened in each transaction, including both ETH and token movements.

## How to Use
1. **Place Data:**
   - Place the competition `.parquet` files in the appropriate folders:
     - `data/base/`
     - `data/ethereum/`
2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Pipeline:**
   - For **Base** dataset (default):
     ```sh
     python main.py
     ```
     or explicitly:
     ```sh
     python main.py base
     ```
     Output: `submission.csv`
   - For **Ethereum** dataset:
     ```sh
     python main.py ethereum
     ```
     Output: `submission_ethereum.csv`

## What the Pipeline Does
- Loads and merges labeled and test addresses, transactions, token transfers, and DEX swaps.
- Engineers basic behavioral features for each address.
- Trains a LightGBM classifier to predict Sybil likelihood.
- Outputs a CSV with `ADDRESS,PRED` for submission.

## Evaluation Metric: AUC (Area Under the ROC Curve)

- **AUC** measures how well your model separates Sybil from non-Sybil addresses.
- **Formula:**
  \[
  AUC = \int_0^1 TPR(FPR^{-1}(x)) dx
  \]
- **TPR (True Positive Rate):**
  \[
  TPR = \frac{True\ Positives}{True\ Positives + False\ Negatives}
  \]
- **FPR (False Positive Rate):**
  \[
  FPR = \frac{False\ Positives}{False\ Positives + True\ Negatives}
  \]
- As you move the classification threshold from 0 to 1, you trace out the ROC curve in the (FPR, TPR) plane. Integrating TPR with respect to FPR gives the AUC.

## Customization
- You can extend `src/feature_engineering.py` to add more sophisticated features, including joining transactions and token transfers using `Tx_Hash`.
- The code is modular and ready for advanced experimentation.

## Requirements
- Python 3.12+
- See `requirements.txt` for required packages (pandas, numpy, scikit-learn, lightgbm, pyarrow, jupyter, etc.)

## Notes
- Make sure the label column in the train set is `LABEL` (0 = non-Sybil, 1 = Sybil).
- Predictions must be made for every address in the test set for a valid submission.
- The pipeline auto-selects the dataset based on the command-line argument (`base` or `ethereum`).

---

For questions or improvements, feel free to open an issue or contribute!
