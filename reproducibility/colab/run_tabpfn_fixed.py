"""Colab driver: TabPFN on S1 only, with the context-sampling bug fixed.

The earlier driver drew max(200, share * 10,000) records per KABCO class and then cut the
concatenated list to 10,000. The fatal class came last, so the cut removed every K record
and TabPFN assigned probability zero to K. This version guarantees the minimum for every
class and takes the remainder from the no-injury class, so the context has exactly
CONTEXT records and contains all five classes.

Run in Colab after mounting Drive (same folder layout as run_all_colab.py):
    %run run_tabpfn_fixed.py
Output: MyDrive/CHOIR/colab_outputs/s1_tabpfn_fixed.npz (rename to s1_tabpfn.npz locally).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import importlib
import sys
if "colab_common" in sys.modules:
    importlib.reload(sys.modules["colab_common"])
import numpy as np
import colab_common as cc

BASE = Path("/content/drive/MyDrive/CHOIR")
SNAPSHOT = BASE / "analysis.parquet"
OUT_DIR = BASE / "colab_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_F = OUT_DIR / "s1_tabpfn_fixed.npz"

CONTEXT = 10_000
MIN_PER_CLASS = 200
EXPECTED_S1_FINGERPRINT = "416bdd0ac68fbfec"   # current snapshot, see RUNBOOK


def cdf_from_proba(proba):
    cdf = np.cumsum(proba, axis=1)
    return (cdf / cdf[:, -1:]).astype(np.float32)


def stratified_context(y_tr, rng):
    """Proportional allocation with a floor; the no-injury class absorbs the remainder."""
    counts = {k: int((y_tr == k).sum()) for k in range(1, 6)}
    n_k = {k: max(MIN_PER_CLASS, int(round(CONTEXT * counts[k] / len(y_tr)))) for k in range(2, 6)}
    n_k[1] = CONTEXT - sum(n_k.values())
    idx = [rng.choice(np.where(y_tr == k)[0], min(n_k[k], counts[k]), replace=False) for k in range(1, 6)]
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx, {k: int((y_tr[idx] == k).sum()) for k in range(1, 6)}


def main():
    df = cc.load_primary(SNAPSHOT)
    tr, ca, te = cc.split_s1(df)[:3]
    fp = cc.split_fingerprint(ca, te)
    print(f"s1: train={len(tr):,} cal={len(ca):,} test={len(te):,} fingerprint {fp}", flush=True)
    if fp != EXPECTED_S1_FINGERPRINT:
        raise SystemExit(f"Fingerprint {fp} != {EXPECTED_S1_FINGERPRINT}: wrong analysis.parquet on Drive. Stop.")

    if not os.environ.get("TABPFN_TOKEN"):
        tok = BASE / "tabpfn_token.txt"
        if tok.exists():
            os.environ["TABPFN_TOKEN"] = tok.read_text().strip()
    print("TABPFN_TOKEN present:", bool(os.environ.get("TABPFN_TOKEN")), flush=True)

    enc = cc.make_encoder().fit(tr)
    X_tr, X_ca, X_te = cc.encode(enc, tr), cc.encode(enc, ca), cc.encode(enc, te)
    y_tr = tr["y_kabco"].to_numpy()

    rng = np.random.default_rng(cc.SEED)
    idx, composition = stratified_context(y_tr, rng)
    print("context composition by KABCO class:", composition, flush=True)
    assert len(idx) == CONTEXT and all(v >= MIN_PER_CLASS for k, v in composition.items() if k > 1)

    from tabpfn import TabPFNClassifier
    import tabpfn
    t = time.time()
    clf = TabPFNClassifier()
    clf.fit(X_tr[idx], y_tr[idx])
    fit_s = time.time() - t
    assert sorted(int(c) for c in clf.classes_) == [1, 2, 3, 4, 5], clf.classes_

    def proba(X, chunk=20_000):
        out = []
        for i in range(0, len(X), chunk):
            out.append(clf.predict_proba(X[i:i + chunk]).astype(np.float32))
            if i % 200_000 == 0:
                print(f"  predict {i:,}/{len(X):,}", flush=True)
        p = np.concatenate(out)
        full = np.zeros((len(p), 5), np.float32)
        for j, c in enumerate(clf.classes_):
            full[:, int(c) - 1] = p[:, j]
        return full / full.sum(1, keepdims=True)

    t = time.time()
    cdf_ca = cdf_from_proba(proba(X_ca))
    cdf_te = cdf_from_proba(proba(X_te))
    predict_s = time.time() - t
    pk = 1 - cdf_te[:, 3]
    print(f"mean P(K) on test: {pk.mean():.5f} (must be > 0)", flush=True)
    np.savez_compressed(OUT_F, cdf_ca=cdf_ca, cdf_te=cdf_te, fit_s=fit_s,
                        predict_s=predict_s, fingerprint=fp)
    (OUT_DIR / "s1_tabpfn_fixed_meta.json").write_text(json.dumps({
        "context": CONTEXT, "min_per_class": MIN_PER_CLASS, "composition": composition,
        "tabpfn_version": getattr(tabpfn, "__version__", "unknown"),
        "fit_s": fit_s, "predict_s": predict_s, "mean_pK_test": float(pk.mean())}, indent=2))
    print(f"DONE fit={fit_s:.0f}s predict={predict_s:.0f}s -> {OUT_F}", flush=True)


if __name__ == "__main__":
    main()
