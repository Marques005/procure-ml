"""Step 3: tune the boosted model, check whether its probabilities can be trusted, and
measure what each model costs to run.

Usage:  python scripts/tune_and_calibrate.py
Writes: results/tuning.md, results/calibration.png, results/model_choice.csv
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (unlocks the class below)
from sklearn.model_selection import HalvingRandomSearchCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from procureml import classify as C  # noqa: E402
from procureml.data import load_world  # noqa: E402

COLORS = {"Logistic regression": "#2a78d6", "Gradient boosting (tuned)": "#1baf7a",
          "Overfit boosting": "#eb6834", "Overfit boosting + isotonic": "#4a3aa7"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.axisbelow": True, "font.size": 10})


def calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: in each probability bin, |mean predicted - observed rate|,
    weighted by how many orders fall in the bin."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return err


def main():
    w = load_world(ROOT / "data")
    history, new = w["history"], w["new"]
    cats = C.categories_of(history)
    X_hist, y_hist = C.as_model_input(history, cats), history[C.TARGET]
    X_new, y_new = C.as_model_input(new, cats), new[C.TARGET].to_numpy()
    truth = new.true_p_late.to_numpy()

    # 1. Hyperparameter search with successive halving.
    space = {"learning_rate": loguniform(0.01, 0.5), "max_leaf_nodes": randint(4, 64),
             "min_samples_leaf": randint(10, 400), "l2_regularization": loguniform(1e-3, 10),
             "max_iter": randint(50, 400)}
    search = HalvingRandomSearchCV(C.make_gbm(), space, n_candidates=48, factor=3,
                                   resource="n_samples", min_resources=2_000,
                                   scoring="neg_log_loss", cv=3, random_state=0, n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(X_hist, y_hist)
    search_s = time.perf_counter() - t0
    rounds = pd.DataFrame(search.cv_results_).groupby("iter").agg(
        candidates=("params", "size"), samples_each=("n_resources", "first"),
        best_log_loss=("mean_test_score", lambda s: -s.max()))
    best = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()}
    print(f"search: {search_s:.0f}s, best params {best}")

    # 2. Models to compare. The overfit one is deliberately bad, to show what calibration fixes.
    overfit = dict(learning_rate=0.3, max_leaf_nodes=63, min_samples_leaf=5, max_iter=300,
                   early_stopping=False)
    models = {
        "Logistic regression": C.make_logistic(),
        "Gradient boosting (tuned)": C.make_gbm(**search.best_params_),
        "Overfit boosting": C.make_gbm(**overfit),
        "Overfit boosting + isotonic": CalibratedClassifierCV(C.make_gbm(**overfit), method="isotonic", cv=3),
    }
    rows, probs = [], {}
    for name, model in models.items():
        p, fit_s, us = C.timed_fit_predict(model, X_hist, y_hist, X_new)
        probs[name] = p
        rows.append({"model": name, **C.scores(y_new, p),
                     "calibration_error": calibration_error(p, y_new),
                     "mean_abs_error_vs_truth": np.abs(p - truth).mean(),
                     "fit_s": fit_s, "predict_us_per_order": us})
        print(f"{name:30s} done")
    rows.append({"model": "True probabilities (ceiling)", **C.scores(y_new, truth),
                 "calibration_error": calibration_error(truth, y_new), "mean_abs_error_vs_truth": 0.0})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(ROOT / "results" / "model_choice.csv")

    # 3. Reliability diagram: predicted probability vs observed late rate, per bin.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="#52514e", ls="--", lw=1, label="perfect calibration")
    edges = np.linspace(0, 1, 11)
    for name, p in probs.items():
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        pts = [(p[idx == b].mean(), y_new[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 50]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=4, lw=2, color=COLORS[name], label=name)
    ax.set_xlabel("predicted probability of being late")
    ax.set_ylabel("observed share of late orders")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration on 20,000 new orders", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "calibration.png", dpi=150)
    plt.close(fig)

    md = ["# Tuning, calibration and cost", "",
          f"## Successive halving ({len(search.cv_results_['params'])} fits in {search_s:.0f}s)", "",
          rounds.round(4).to_markdown(), "", f"Best parameters: `{best}`", "",
          "## Models compared on new orders", "", table.round(4).to_markdown(), ""]
    (ROOT / "results" / "tuning.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
