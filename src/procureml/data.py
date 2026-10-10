"""Build the P2 world: order history, new orders and supplier quotes, all from P1's simulator.

Two kinds of columns:
  * what a buyer could know (features, past outcomes, supplier price quotes)
  * the hidden truth, prefixed `true_` (only the simulator knows it). Models never
    see these columns; they are only used to grade the models afterwards.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from synthsupply.generators import DomainSimulator
from synthsupply.prepare import load_reference, split

SEED = 42
N_HISTORY = 80_000        # past orders with known outcomes: what models learn from
N_NEW = 20_000            # new orders: what we predict and recommend suppliers for
LATE_PENALTY = 0.25       # a late delivery costs 25% of the order value (stoppage, expediting)


def build_world(dataco_csv: str | Path, out_dir: str | Path, seed: int = SEED) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. The same simulator as P1: calibrated on DataCo's training part.
    train, _ = split(load_reference(dataco_csv), seed=seed)
    sim = DomainSimulator(seed=seed).fit(train)

    # 2. History, then new orders (two independent draws from the same world).
    history = sim.sample(N_HISTORY)
    new = sim.sample(N_NEW)
    new["true_p_late"] = sim.late_probability(new)   # call right after sampling these rows
    for df in (history, new):
        df["order_value"] = (df.unit_price * df.quantity * (1 - df.discount_rate)).round(2)

    # 3. Supplier quotes. Reliable suppliers charge more: that is the buyer's trade-off.
    rng = np.random.default_rng(seed + 7)
    suppliers = pd.DataFrame([
        {"supplier_id": s, "department": dept, "true_risk": sim.supplier_risk[s]}
        for dept, ss in sim.suppliers.items() for s in ss
    ])
    suppliers["price_premium"] = np.clip(
        0.04 - 0.03 * suppliers.true_risk + rng.normal(0, 0.01, len(suppliers)), -0.03, 0.12
    ).round(4)

    history.to_csv(out / "history.csv", index=False)
    new.to_csv(out / "new_orders.csv", index=False)
    suppliers.to_csv(out / "suppliers.csv", index=False)
    with open(out / "simulator.pkl", "wb") as f:      # kept to compute the truth for any supplier
        pickle.dump(sim, f)
    return {"history": history, "new": new, "suppliers": suppliers, "sim": sim}


def load_world(data_dir: str | Path) -> dict:
    d = Path(data_dir)
    with open(d / "simulator.pkl", "rb") as f:
        sim = pickle.load(f)
    return {"history": pd.read_csv(d / "history.csv"), "new": pd.read_csv(d / "new_orders.csv"),
            "suppliers": pd.read_csv(d / "suppliers.csv"), "sim": sim}
