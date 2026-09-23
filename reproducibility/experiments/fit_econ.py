#!/usr/bin/env python
"""Fit the econometric-canon anchors on S1 and cache their CDFs (+ LC classes).

Outputs: cache/s1_lc_logit.npz, cache/s1_rp_logit.npz (standard cache format),
cache/s1_lc_classes.npz (classical partition for E3).
"""
import time

import numpy as np

from common import (SEED, CACHE, load_primary, split_s1, make_encoder, encode,
                    _split_fingerprint)
from econ_models import LCOrderedLogit, RPOrderedLogit
from choir import cdf_from_proba


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    enc = make_encoder().fit(tr)
    X_tr, X_ca, X_te = encode(enc, tr), encode(enc, ca), encode(enc, te)
    y_tr = tr["y_kabco"].to_numpy()
    # Row-order fingerprint of THIS split. common.py::prepared_cdfs refuses any cache
    # without a matching "fingerprint" key, so every npz written here must carry it.
    fp = _split_fingerprint(ca, te)
    print(f"setup {time.time()-t0:.0f}s; s1 fingerprint {fp}", flush=True)

    for model in (LCOrderedLogit(n_classes=3), RPOrderedLogit(rp_col=0)):
        f = CACHE / f"s1_{model.name}.npz"
        if f.exists():
            print(f"{model.name}: cached, skip", flush=True)
            continue
        t = time.time()
        model.fit(X_tr, y_tr)
        fit_s = time.time() - t
        t = time.time()
        cdf_ca = cdf_from_proba(model.predict_proba(X_ca)).astype(np.float32)
        cdf_te = cdf_from_proba(model.predict_proba(X_te)).astype(np.float32)
        predict_s = time.time() - t
        np.savez_compressed(f, cdf_ca=cdf_ca, cdf_te=cdf_te,
                            fit_s=fit_s, predict_s=predict_s, fingerprint=fp)
        print(f"{model.name}: fit {fit_s:.0f}s predict {predict_s:.0f}s", flush=True)
        if model.name == "lc_logit":
            np.savez_compressed(CACHE / "s1_lc_classes.npz",
                                cls_ca=model.predict_classes(X_ca),
                                cls_te=model.predict_classes(X_te), fingerprint=fp)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
