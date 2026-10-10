"""Step 4: recommend a supplier for each new order and grade the choices against the truth.

Usage:  python scripts/recommend_suppliers.py
Writes: results/recommendation.md, results/recommendation.csv, results/penalty_sensitivity.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from procureml import classify as C  # noqa: E402
from procureml import recommend as R  # noqa: E402
from procureml.data import LATE_PENALTY, load_world  # noqa: E402

COLORS = {"Random supplier": "#52514e", "Cheapest quote": "#eb6834",
          "Lowest past late rate": "#2a78d6", "Model expected cost": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.axisbelow": True, "font.size": 10})


def strategies_for(cand, p_model, past_rate, penalty, rng):
    return {
        "Random supplier": rng.random(len(cand)),
        "Cheapest quote": cand.price_premium.to_numpy(),
        "Lowest past late rate": cand.supplier_id.map(past_rate).to_numpy(),
        "Model expected cost": R.expected_cost(cand, p_model, penalty).to_numpy(),
    }


def main():
    w = load_world(ROOT / "data")
    history, new, suppliers, sim = w["history"], w["new"], w["suppliers"], w["sim"]
    cats = C.categories_of(history)

    # The model chosen in step 3: logistic regression (as good as boosting, better calibrated, faster).
    model = C.make_logistic().fit(C.as_model_input(history, cats), history[C.TARGET])

    cand = R.candidates(new, suppliers)
    p_model = R.model_p_late(model, cand, cats)
    true_p = sim.late_probability(cand)                    # hidden truth, used only for grading
    past_rate = history.groupby("supplier_id").late.mean()  # what a buyer sees in the history
    print(f"{cand.order_id.nunique():,} orders x suppliers = {len(cand):,} candidates")

    rng = np.random.default_rng(0)
    table = R.evaluate(cand, strategies_for(cand, p_model, past_rate, LATE_PENALTY, rng), true_p, LATE_PENALTY)
    table.to_csv(ROOT / "results" / "recommendation.csv")

    # How does the answer change with the cost of a delay? (an assumption, so test it)
    penalties = [0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6]
    sens = {}
    for pen in penalties:
        t = R.evaluate(cand, strategies_for(cand, p_model, past_rate, pen, np.random.default_rng(0)), true_p, pen)
        sens[pen] = t.regret_pct_of_value
    sens = pd.DataFrame(sens).T
    sens.index.name = "late_penalty"
    sens.to_csv(ROOT / "results" / "penalty_sensitivity.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in sens.columns:
        ax.plot(sens.index * 100, sens[name], marker="o", ms=4, lw=2, color=COLORS[name], label=name)
    ax.axvline(LATE_PENALTY * 100, color="#52514e", ls="--", lw=1)
    ax.text(LATE_PENALTY * 100 + 0.8, ax.get_ylim()[1] * 0.92, "assumed cost", fontsize=8, color="#52514e")
    ax.set_xlabel("cost of a late delivery (% of order value)")
    ax.set_ylabel("regret vs best choice (% of value)")
    ax.set_title("Only the model stays close to the best supplier at every delay cost", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "penalty_sensitivity.png", dpi=150)
    plt.close(fig)

    md = ["# Supplier recommendation", "",
          f"20,000 new orders, 3 candidate suppliers each. Late penalty = {LATE_PENALTY:.0%} of order value.",
          "Every choice is graded with the simulator's true probabilities.", "",
          table.round(4).to_markdown(), "",
          "## Regret (% of order value) for different costs of a delay", "",
          sens.round(3).to_markdown(), ""]
    (ROOT / "results" / "recommendation.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
