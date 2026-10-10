"""Late-delivery classifiers: what a buyer can know when placing the order -> P(late).

Only information available AT ORDER TIME is used. days_real and late are outcomes
(known only after delivery), and every `true_` column is hidden truth: none of them
are features.
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["shipping_mode", "market", "customer_segment", "department", "payment_type",
               "supplier_id", "order_month"]          # month as a category: the peak is Nov/Dec only
NUMERIC = ["quantity", "unit_price", "discount_rate", "order_dow"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "late"


def as_model_input(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Feature table with categorical columns typed as pandas categories (same codes everywhere)."""
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def categories_of(df: pd.DataFrame) -> dict:
    return {c: sorted(df[c].unique()) for c in CATEGORICAL}


def make_logistic():
    """Baseline: one weight per category value + a weight per (scaled) number, added up."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def make_gbm(**params):
    """Gradient boosting: trees added one after another, each correcting the previous ones."""
    return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0, **params)


class ModeRateBaseline:
    """What a buyer would do with a spreadsheet: the historical late rate of the shipping mode."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.rates = y.groupby(X.shipping_mode.astype(str)).mean()
        self.default = y.mean()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X.shipping_mode.astype(str).map(self.rates).fillna(self.default).to_numpy()
        return np.column_stack([1 - p, p])


def scores(y: pd.Series, p: np.ndarray) -> dict:
    """AUC = ranking quality; Brier and log loss = quality of the probabilities themselves."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}


def timed_fit_predict(model, X_train, y_train, X_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = model.predict_proba(X_test)[:, 1]
    per_order_us = (time.perf_counter() - t0) / len(X_test) * 1e6
    return p, fit_s, per_order_us
