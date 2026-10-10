"""Supplier recommendation: for each new order, rank the department's suppliers.

expected cost(supplier) = order value x (1 + price premium)            what we pay
                        + LATE_PENALTY x order value x P(late | supplier)  what a delay costs us

Every candidate is the same order with only the supplier changed. A strategy picks one;
it is graded with the TRUE probabilities from the simulator, which no strategy sees.
"""

import numpy as np
import pandas as pd

from . import classify as C


def candidates(new: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    """One row per (order, possible supplier): the order copied once per supplier of its department."""
    orders = new.drop(columns=["supplier_id", "true_p_late"]).reset_index(names="order_id")
    cand = orders.merge(suppliers[["supplier_id", "department", "price_premium"]], on="department")
    return cand.sort_values(["order_id", "supplier_id"]).reset_index(drop=True)


def expected_cost(cand: pd.DataFrame, p_late: np.ndarray, penalty: float) -> np.ndarray:
    return cand.order_value * (1 + cand.price_premium) + penalty * cand.order_value * p_late


def pick(cand: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """Index of the candidate with the LOWEST score in each order."""
    return pd.Series(score, index=cand.index).groupby(cand.order_id).idxmin()


def ndcg(cand: pd.DataFrame, score: np.ndarray, true_cost: np.ndarray) -> float:
    """Ranking quality vs the true ranking: relevance 2 / 1 / 0 for the truly best / 2nd / 3rd
    supplier; NDCG = 1 when the strategy orders all three exactly as the truth does."""
    df = pd.DataFrame({"order_id": cand.order_id, "score": score, "true": true_cost})
    df["rel"] = df.groupby("order_id")["true"].rank(method="first", ascending=False) - 1  # best -> 2
    df["pos"] = df.groupby("order_id")["score"].rank(method="first")                      # 1 = picked first
    df["gain"] = df.rel / np.log2(df.pos + 1)
    df["ideal_pos"] = df.groupby("order_id")["true"].rank(method="first")
    df["ideal_gain"] = df.rel / np.log2(df.ideal_pos + 1)
    g = df.groupby("order_id")[["gain", "ideal_gain"]].sum()
    return float((g.gain / g.ideal_gain).mean())


def evaluate(cand: pd.DataFrame, strategies: dict, true_p: np.ndarray, penalty: float) -> pd.DataFrame:
    true_cost = expected_cost(cand, true_p, penalty).to_numpy()
    best = pick(cand, true_cost)
    value = cand.loc[best, "order_value"].to_numpy()
    rows = []
    for name, score in strategies.items():
        chosen = pick(cand, score)
        regret = true_cost[chosen.to_numpy()] - true_cost[best.to_numpy()]
        rows.append({"strategy": name,
                     "hit_at_1": float((chosen.to_numpy() == best.to_numpy()).mean()),
                     "ndcg": ndcg(cand, score, true_cost),
                     "true_late_rate": float(true_p[chosen.to_numpy()].mean()),
                     "regret_pct_of_value": float(100 * regret.sum() / value.sum()),
                     "total_cost": float(true_cost[chosen.to_numpy()].sum())})
    return pd.DataFrame(rows).set_index("strategy")


def model_p_late(model, cand: pd.DataFrame, cats: dict) -> np.ndarray:
    return model.predict_proba(C.as_model_input(cand, cats))[:, 1]
