import pandas as pd

def make_predictions(model, X_test: pd.DataFrame, addresses: pd.Series, out_path: str):
    """
    Generate predictions and write to CSV in the required format.
    """
    preds = model.predict_proba(X_test)[:, 1]
    df_out = pd.DataFrame({'ADDRESS': addresses, 'PRED': preds})
    df_out.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")
