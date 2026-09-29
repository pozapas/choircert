"""Alternative conformal baselines on the primary sample (S1).

Uses the cached base-model probabilities; nothing is refit. For each base model:
  - LAC marginal: score 1 - p(y|x), one threshold.
  - Label-conditional LAC: one threshold per KABCO label (class-conditional conformal).
  - APS (non-randomized): cumulative mass of labels ranked by probability.
  - CHOIR marginal and CHOIR four-cell (ordinal CDF score), for reference.
Reports overall and per-stratum coverage, mean set size, and the share of sets that
are contiguous on the KABCO scale.
"""
import json
import numpy as np
import pandas as pd
from common import SEED, RESULTS, load_primary, split_s1, prepared_cdfs
from e2_e3_heterogeneity import declared_partition
from choir import cumulative_score, interval_sets, split_calibrate, mondrian_calibrate

ALPHA = 0.10
MODELS = ["histgb", "ordered_logit", "dlcon"]


def conf_q(scores, alpha):
    n = len(scores)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def probs_from_cdf(cdf):
    c = np.concatenate([np.zeros((len(cdf), 1), dtype=cdf.dtype), cdf], axis=1)
    return np.clip(np.diff(c, axis=1), 0, 1)


def set_stats(member, y, cells):
    n, K = member.shape
    cov = member[np.arange(n), y - 1]
    size = member.sum(1)
    # contiguous: included labels form one run (empty sets counted as contiguous)
    diff = np.diff(np.concatenate([np.zeros((n, 1), bool), member, np.zeros((n, 1), bool)], 1).astype(int), axis=1)
    runs = (diff == 1).sum(1)
    contig = runs <= 1
    out = {"coverage": float(cov.mean()), "mean_size": float(size.mean()),
           "contiguous_share": float(contig.mean()), "empty_share": float((size == 0).mean())}
    for c in np.unique(cells):
        m = cells == c
        out[f"cov_{c}"] = float(cov[m].mean())
        out[f"size_{c}"] = float(size[m].mean())
    return out


def main():
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca = ca["y_kabco"].to_numpy().astype(int)
    y_te = te["y_kabco"].to_numpy().astype(int)
    cell_ca, cell_te = declared_partition(ca), declared_partition(te)
    rows = []
    for m in MODELS:
        cdf_ca, cdf_te, _ = prepared_cdfs("s1", m, tr, ca, te, X=(None, None, None))
        p_ca, p_te = probs_from_cdf(cdf_ca), probs_from_cdf(cdf_te)
        n_ca = len(y_ca)
        # LAC marginal
        s = 1 - p_ca[np.arange(n_ca), y_ca - 1]
        q = conf_q(s, ALPHA)
        rows.append({"model": m, "method": "LAC marginal", **set_stats((1 - p_te) <= q, y_te, cell_te)})
        # Label-conditional LAC
        qk = np.array([conf_q(s[y_ca == k], ALPHA) for k in range(1, 6)])
        rows.append({"model": m, "method": "Label-conditional LAC", **set_stats((1 - p_te) <= qk[None, :], y_te, cell_te)})
        # APS (non-randomized)
        order = np.argsort(-p_ca, axis=1)
        ranks = np.argsort(order, axis=1)
        srt = np.take_along_axis(p_ca, order, 1)
        cums = np.cumsum(srt, 1)
        s_aps = cums[np.arange(n_ca), ranks[np.arange(n_ca), y_ca - 1]]
        qa = conf_q(s_aps, ALPHA)
        o_te = np.argsort(-p_te, axis=1)
        c_te = np.cumsum(np.take_along_axis(p_te, o_te, 1), 1)
        prev = np.concatenate([np.zeros((len(p_te), 1)), c_te[:, :-1]], 1)
        inc_sorted = prev < qa  # include label while mass before it is below q
        member = np.zeros_like(inc_sorted)
        np.put_along_axis(member, o_te, inc_sorted, 1)
        rows.append({"model": m, "method": "APS", **set_stats(member, y_te, cell_te)})
        # CHOIR marginal and four-cell
        sc = cumulative_score(cdf_ca, y_ca)
        qm = split_calibrate(sc, ALPHA)
        lo, hi = interval_sets(cdf_te, qm)
        mem = (np.arange(1, 6)[None, :] >= lo[:, None]) & (np.arange(1, 6)[None, :] <= hi[:, None])
        rows.append({"model": m, "method": "CHOIR marginal", **set_stats(mem, y_te, cell_te)})
        qc = mondrian_calibrate(sc, cell_ca, ALPHA)
        lam = np.array([qc[c] for c in cell_te])
        lo, hi = interval_sets(cdf_te, lam)
        mem = (np.arange(1, 6)[None, :] >= lo[:, None]) & (np.arange(1, 6)[None, :] <= hi[:, None])
        rows.append({"model": m, "method": "CHOIR four-cell", **set_stats(mem, y_te, cell_te)})
        print(m, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "b1_baselines.parquet", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
