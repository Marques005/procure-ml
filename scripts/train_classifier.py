"""Step 2: compare late-delivery classifiers on new orders, against the true probabilities.

Usage:  python scripts/train_classifier.py
Writes: results/classifiers.md, results/classifiers.csv, results/importance.csv
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from procureml import classify as C  # noqa: E402
from procureml.data import load_world  # noqa: E402


def main():
    w = load_world(ROOT / "data")
    history, new = w["history"], w["new"]
    cats = C.categories_of(history)
    X_hist, y_hist = C.as_model_input(history, cats), history[C.TARGET]
    X_new, y_new = C.as_model_input(new, cats), new[C.TARGET]

    models = {"Mode late rate (spreadsheet)": C.ModeRateBaseline(),
              "Logistic regression": C.make_logistic(),
              "Gradient boosting (defaults)": C.make_gbm()}
    rows, fitted = [], {}
    for name, model in models.items():
        p, fit_s, us = C.timed_fit_predict(model, X_hist, y_hist, X_new)
        fitted[name] = model
        rows.append({"model": name, **C.scores(y_new, p), "fit_s": fit_s, "predict_us_per_order": us})
        print(f"{name:30s} done")
    # The ceiling: the simulator's own probabilities. No model can beat these on average.
    rows.append({"model": "True probabilities (ceiling)", **C.scores(y_new, new.true_p_late.to_numpy())})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(ROOT / "results" / "classifiers.csv")

    # Which inputs does the boosted model rely on? Shuffle one column, see how much AUC drops.
    imp = permutation_importance(fitted["Gradient boosting (defaults)"], X_new, y_new,
                                 scoring="roc_auc", n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, index=C.FEATURES).sort_values(ascending=False)
    importance.to_csv(ROOT / "results" / "importance.csv", header=["auc_drop"])

    md = ["# Late-delivery classifiers", "",
          "Trained on 80,000 past orders, scored on 20,000 new ones.", "",
          table.round(4).to_markdown(), "",
          "## Permutation importance (gradient boosting)", "",
          "AUC lost when one input is shuffled. Large = the model relies on it.", "",
          importance.round(4).to_frame("auc_drop").to_markdown(), ""]
    (ROOT / "results" / "classifiers.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
