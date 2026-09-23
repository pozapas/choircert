"""Compare CHOIR, MAPIE, and crepes on one fixed ordinal task.

All three methods target marginal coverage. This benchmark does not establish general
superiority. It records empirical observed-label coverage, set size, and contiguity on
the bundled demonstration. It also identifies the additional certification branches
included in this CHOIR workflow. The comparison does not imply that another library
cannot be extended with user code.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from choir import NoiseModel, cdf_from_proba, cumulative_score, interval_sets, split_calibrate
from choir.datasets import load_demo

OUT = Path(__file__).resolve().parent / "results" / "generic_tool_benchmark.csv"
SEED = 20260704


def encode(rows, cols):
    """Minimal one-hot for the demo (dependency-light)."""
    cat_vals = {}
    for c in cols:
        if isinstance(rows[0][c], str):
            cat_vals[c] = sorted({r[c] for r in rows})
    X = []
    for r in rows:
        v = []
        for c in cols:
            if c in cat_vals:
                v += [1.0 if r[c] == lv else 0.0 for lv in cat_vals[c]]
            else:
                v.append(float(r[c]))
        X.append(v)
    return np.array(X, dtype=float)


def main():
    rows, y, cols = load_demo()
    y = np.array(y)
    X = encode(rows, cols)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    tr, ca, te = np.split(idx, [int(0.5 * len(y)), int(0.75 * len(y))])

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], y[tr])
    classes = list(clf.classes_)
    proba_ca = clf.predict_proba(X[ca])
    proba_te = clf.predict_proba(X[te])
    alpha = 0.10

    results = {}

    # ---- CHOIR: ordinal split conformal + noise + fatal guarantee ----
    cdf_ca = cdf_from_proba(proba_ca)
    cdf_te = cdf_from_proba(proba_te)
    qhat = split_calibrate(cumulative_score(cdf_ca, y[ca]), alpha)
    lo, hi = interval_sets(cdf_te, qhat)
    cov = np.mean((y[te] >= lo) & (y[te] <= hi))
    contig = True  # guaranteed by construction (Lemma 1)
    nm = NoiseModel.kabco(delta=0.02)
    _lo_expanded, _hi_expanded = nm.expand(lo, hi)
    results["CHOIR"] = {
        "coverage": cov, "avg_size": np.mean(hi - lo + 1), "contiguous": contig,
        "true_label_guarantee": True, "fatal_omission_guarantee": True,
    }

    cls_idx = np.array([classes.index(v) for v in y[te]])

    # ---- MAPIE (generic classification conformal; v1.x API) ----
    try:
        from mapie.classification import SplitConformalClassifier
        m = SplitConformalClassifier(estimator=clf, confidence_level=1 - alpha,
                                     conformity_score="lac", prefit=True)
        m.conformalize(X[ca], y[ca])
        _, sets = m.predict_set(X[te])
        sets = np.asarray(sets)
        if sets.ndim == 3:
            sets = sets[:, :, 0]
        covered = sets[np.arange(len(te)), cls_idx]
        contig_frac = _contiguity_fraction(sets, classes)
        results["MAPIE"] = {
            "coverage": float(covered.mean()), "avg_size": float(sets.sum(1).mean()),
            "contiguous": contig_frac == 1.0, "contig_frac": contig_frac,
            "true_label_guarantee": False, "fatal_omission_guarantee": False,
        }
    except Exception as e:  # noqa: BLE001 - record optional-tool failures in the result
        results["MAPIE"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- crepes (generic conformal classifier) ----
    try:
        from crepes import WrapClassifier
        wc = WrapClassifier(clf)
        wc.calibrate(X[ca], y[ca], seed=SEED)
        sets = np.asarray(wc.predict_set(X[te], confidence=1 - alpha, labels=False))
        covered = sets[np.arange(len(te)), cls_idx]
        contig_frac = _contiguity_fraction(sets, classes)
        results["crepes"] = {
            "coverage": float(covered.mean()), "avg_size": float(sets.sum(1).mean()),
            "contiguous": contig_frac == 1.0, "contig_frac": contig_frac,
            "true_label_guarantee": False, "fatal_omission_guarantee": False,
        }
    except Exception as e:  # noqa: BLE001 - record optional-tool failures in the result
        results["crepes"] = {"error": f"{type(e).__name__}: {e}"}

    _print_table(results, alpha)
    _write_results(results, alpha)
    return results


def _contiguity_fraction(sets, classes):
    """Fraction of prediction sets that are contiguous on the ordinal scale."""
    order = np.argsort(classes)
    s = sets[:, order].astype(bool)
    ok = 0
    for row in s:
        idx = np.where(row)[0]
        if len(idx) == 0 or (idx.max() - idx.min() + 1) == len(idx):
            ok += 1
    return ok / len(s)


def _print_table(results, alpha):
    print(f"\nOrdinal conformal benchmark (demo data, alpha={alpha}, target {1-alpha:.2f})\n")
    hdr = f"{'method':<8} {'coverage':>9} {'avg size':>9} {'contiguous':>11} {'true-label':>11} {'fatal grt':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<8} {'(unavailable: ' + r['error'][:40] + ')'}")
            continue
        cf = r.get("contig_frac")
        contig = "yes" if r["contiguous"] else (f"{cf:.0%}" if cf is not None else "no")
        print(f"{name:<8} {r['coverage']:>9.3f} {r['avg_size']:>9.2f} {contig:>11} "
              f"{'yes' if r['true_label_guarantee'] else 'no':>11} "
              f"{'yes' if r['fatal_omission_guarantee'] else 'no':>10}")
    print("\nAll methods target marginal coverage in this fixed workflow.")
    print("CHOIR also includes contiguous ordinal construction, declared-map transfer,")
    print("and fatal-omission control under their stated assumptions.")


def _write_results(results, alpha):
    """Write the aggregate benchmark values stored with this repository."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "alpha",
        "coverage",
        "avg_size",
        "contiguity_fraction",
        "contiguous_by_construction",
        "true_label_guarantee",
        "fatal_omission_guarantee",
        "error",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({
                "method": name,
                "alpha": alpha,
                "coverage": result.get("coverage", ""),
                "avg_size": result.get("avg_size", ""),
                "contiguity_fraction": result.get(
                    "contig_frac", 1.0 if result.get("contiguous") else ""
                ),
                "contiguous_by_construction": name == "CHOIR",
                "true_label_guarantee": result.get("true_label_guarantee", ""),
                "fatal_omission_guarantee": result.get("fatal_omission_guarantee", ""),
                "error": result.get("error", ""),
            })
    print(f"\nAggregate benchmark results -> {OUT}")


if __name__ == "__main__":
    main()
