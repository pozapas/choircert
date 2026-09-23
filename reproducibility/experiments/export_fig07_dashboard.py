#!/usr/bin/env python
"""Export the three locked Figure 7 deployment-monitoring tables."""
from pathlib import Path
import numpy as np
import pandas as pd

from common import SEED, load_primary, split_s2, prepared_cdfs
from choir import cumulative_score, interval_sets, split_calibrate, mondrian_calibrate

ALPHA = 0.10
N_MIN = 1000
OUT = Path(__file__).resolve().parent.parent / "paper" / "figure_src" / "data"


def month_key(df):
    return df.crash_year.astype(int).astype(str) + "-" + df.crash_month.astype(int).astype(str).str.zfill(2)


def main():
    df = load_primary()
    df["month"] = month_key(df)
    tr, ca, te = split_s2(df)
    cdf_ca, cdf_te, timing = prepared_cdfs("s2", "histgb", tr, ca, te)
    y_ca = ca.y_kabco.to_numpy()
    y_te = te.y_kabco.to_numpy()
    s_ca = cumulative_score(cdf_ca, y_ca)
    s_te = cumulative_score(cdf_te, y_te)

    qhat = split_calibrate(s_ca, ALPHA)
    lo, hi = interval_sets(cdf_te, qhat)
    covered = (y_te >= lo) & (y_te <= hi)

    cnty_ca = ca.Cnty_ID.astype(str).to_numpy()
    cnty_te = te.Cnty_ID.astype(str).to_numpy()
    counts = pd.Series(cnty_ca).value_counts()
    big = set(counts[counts >= N_MIN].index)
    cells_ca = np.where(pd.Series(cnty_ca).isin(big).to_numpy(), cnty_ca, "STATE")
    cells_te = np.where(pd.Series(cnty_te).isin(big).to_numpy(), cnty_te, "STATE")
    q = mondrian_calibrate(s_ca, cells_ca, ALPHA)
    if "STATE" not in q:
        q["STATE"] = qhat
    lam = np.array([q.get(c, q["STATE"]) for c in cells_te])
    lo_m, hi_m = interval_sets(cdf_te, lam)
    covered_m = (y_te >= lo_m) & (y_te <= hi_m)

    te2 = te[["month", "speed_limit"]].copy()
    te2["covered"] = covered
    te2["covered_mondrian"] = covered_m
    monthly = te2.groupby("month", observed=True).agg(
        n=("covered", "size"), cov_unw=("covered", "mean"),
        cov_mondrian=("covered_mondrian", "mean")).reset_index()
    monthly["se"] = np.sqrt(monthly.cov_unw * (1-monthly.cov_unw) / monthly.n)

    all_month = df.groupby("month", observed=True).agg(
        n=("y_kabco", "size"), share_ka=("y_kabco", lambda x: np.mean(x >= 4))).reset_index()
    monthly = all_month.merge(monthly.drop(columns="n"), on="month", how="left")
    monthly.to_csv(OUT / "fig07_monthly.csv", index=False)

    bins = [-np.inf, 34, 44, 54, 64, 69, np.inf]
    labels = ["<35", "35–44", "45–54", "55–64", "65–69", "≥70"]
    te2["speed_band"] = pd.cut(te2.speed_limit, bins=bins, labels=labels)
    heat = te2.groupby(["month", "speed_band"], observed=False).agg(
        n=("covered", "size"), cov=("covered", "mean")).reset_index()
    heat["se"] = np.sqrt(heat["cov"] * (1-heat["cov"]) / heat["n"])
    heat["deviation"] = heat["cov"] - 0.90
    heat["beyond_2se"] = heat.deviation.abs() > 2*heat.se
    heat.to_csv(OUT / "fig07_heat.csv", index=False)

    # Exact smoothed conformal ranks, vectorized to avoid O(n_cal*n_test).
    rng = np.random.default_rng(SEED)
    ss = np.sort(s_ca)
    left = np.searchsorted(ss, s_te, side="left")
    right = np.searchsorted(ss, s_te, side="right")
    gt = len(ss) - right
    eq = right - left
    p = (gt + rng.uniform(size=len(s_te)) * (eq + 1)) / (len(ss) + 1)
    increments = np.log(0.5) - 0.5*np.log(np.clip(p, 1e-12, 1.0))
    log10_m = np.cumsum(increments) / np.log(10)
    mart = pd.DataFrame({"t": np.arange(1, len(te)+1), "month": te.month.to_numpy(),
                         "p_value": p, "log10_M": log10_m})
    mart.to_csv(OUT / "fig07_martingale.csv", index=False)

    probs = [0.05, 0.25, 0.50, 0.75, 0.95]
    uniformity = mart.groupby("month", sort=True)["p_value"].quantile(probs).unstack()
    uniformity.columns = [f"p{int(100*x):02d}" for x in probs]
    uniformity = uniformity.reset_index()
    uniformity = uniformity.merge(monthly[["month", "n"]], on="month", how="left")
    for x in probs:
        col = f"p{int(100*x):02d}"
        uniformity[f"dev_{col}"] = uniformity[col] - x
        uniformity[f"se_{col}"] = np.sqrt(x * (1-x) / uniformity["n"])
    uniformity.to_csv(OUT / "fig07_uniformity.csv", index=False)

    # Restart wealth at one for each natural monthly audit cycle; record its
    # within-month maximum for a compact Ville-valid lollipop panel.
    inc = pd.Series(increments, index=mart.index)
    mart["log10_M_month"] = inc.groupby(mart["month"], sort=False).cumsum() / np.log(10)
    monthly_mart = mart.groupby("month", sort=True).agg(
        max_log10_M=("log10_M_month", "max"),
        end_log10_M=("log10_M_month", "last"), n=("t", "size")).reset_index()

    # Attribution experiment: compare each deployment month only with calibration
    # scores from the same calendar month in 2022–23. Keep the same stream order
    # and smoothed-rank construction; restart wealth at each deployment month.
    cal_month_num = ca.crash_month.astype(int).to_numpy()
    test_month_num = te.crash_month.astype(int).to_numpy()
    p_matched = np.empty(len(te), dtype=float)
    rng_matched = np.random.default_rng(SEED)
    for month_num in range(1, 13):
        cal_m = np.sort(s_ca[cal_month_num == month_num])
        idx = np.where(test_month_num == month_num)[0]
        scores_m = s_te[idx]
        left_m = np.searchsorted(cal_m, scores_m, side="left")
        right_m = np.searchsorted(cal_m, scores_m, side="right")
        gt_m = len(cal_m) - right_m
        eq_m = right_m - left_m
        p_matched[idx] = (gt_m + rng_matched.uniform(size=len(idx)) * (eq_m + 1)) / (len(cal_m) + 1)
    increments_matched = np.log(0.5) - 0.5*np.log(np.clip(p_matched, 1e-12, 1.0))
    mart["p_value_matched"] = p_matched
    mart["log10_M_month_matched"] = (
        pd.Series(increments_matched, index=mart.index)
        .groupby(mart["month"], sort=False).cumsum() / np.log(10)
    )
    matched_monthly = mart.groupby("month", sort=True).agg(
        max_log10_M_matched=("log10_M_month_matched", "max"),
        end_log10_M_matched=("log10_M_month_matched", "last")).reset_index()
    monthly_mart = monthly_mart.merge(matched_monthly, on="month", how="left")
    monthly_mart["pooled_alarm"] = monthly_mart.max_log10_M >= 2.0
    monthly_mart["matched_alarm"] = monthly_mart.max_log10_M_matched >= 2.0
    monthly_mart.to_csv(OUT / "fig07_martingale_monthly.csv", index=False)

    e5 = pd.read_parquet(Path(__file__).resolve().parent / "results" / "e5_temporal.parquet")
    cert = e5[(e5["method"] == "density_ratio_4b") & (e5["year"] == 2025)].iloc[0]
    delta = 0.02
    cert_row = pd.DataFrame([{
        "year": 2025, "level": "state", "nominal": 1-ALPHA,
        "band_delta_declared": delta, "tv_slack_lcb": float(cert.tv_lcb),
        "pre_delta_floor": float(cert.cert_floor_lcb),
        "composed_floor": float(cert.cert_floor_lcb - delta),
        "observed_coverage": float(cert.coverage), "method": "density_ratio_4b",
    }])
    cert_row.to_csv(OUT / "fig07_certificate.csv", index=False)

    dep = monthly.dropna(subset=["cov_unw", "cov_mondrian"])
    mad_unw = float((dep["cov_unw"] - 0.90).abs().mean())
    mad_mon = float((dep["cov_mondrian"] - 0.90).abs().mean())

    print({"cache": timing, "qhat": qhat, "monthly_rows": len(monthly),
           "heat_rows": len(heat), "martingale_rows": len(mart),
           "coverage_range": [float(heat["cov"].min()), float(heat["cov"].max())],
           "log10_M_range": [float(log10_m.min()), float(log10_m.max())],
           "full_stream_sup_t": int(np.argmax(log10_m) + 1),
           "max_monthly_departure": float(uniformity.filter(like="dev_").abs().to_numpy().max()),
           "monthly_restart_max": float(monthly_mart.max_log10_M.max()),
           "pooled_alarm_months": monthly_mart.loc[monthly_mart.pooled_alarm, "month"].tolist(),
           "matched_restart_max": float(monthly_mart.max_log10_M_matched.max()),
           "matched_alarm_months": monthly_mart.loc[monthly_mart.matched_alarm, "month"].tolist(),
           "coverage_mean_abs_dev": {"unweighted": mad_unw, "county_mondrian": mad_mon}})


if __name__ == "__main__":
    main()
