import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a LightGBM model and returns the model and validation AUC.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    return model, auc
