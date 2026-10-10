"""Step 1: generate the P2 world from P1's simulator.

Usage:  python scripts/make_data.py [path/to/DataCoSupplyChainDataset.csv]
Writes: data/history.csv, data/new_orders.csv, data/suppliers.csv, data/simulator.pkl
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from procureml.data import build_world  # noqa: E402

DEFAULT_CSV = ROOT.parent / "synth-supply" / "data" / "DataCoSupplyChainDataset.csv"


def main(csv: Path):
    w = build_world(csv, ROOT / "data")
    h, n, s = w["history"], w["new"], w["suppliers"]
    print(f"history     : {len(h):,} orders, late rate {h.late.mean():.1%}")
    print(f"new orders  : {len(n):,} orders, late rate {n.late.mean():.1%}, "
          f"mean true P(late) {n.true_p_late.mean():.1%}")
    print(f"suppliers   : {len(s)} across {s.department.nunique()} departments")
    print(f"premiums    : {s.price_premium.min():+.1%} to {s.price_premium.max():+.1%}; "
          f"correlation with true risk {s.price_premium.corr(s.true_risk):+.2f}")
    print("\nfirst rows of the history:")
    print(h.head().to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV)
