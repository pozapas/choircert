"""Rebuild the revised result tables from the stored result files (run from the reproducibility folder)."""
import json
import pandas as pd

from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE / "tables"
RES = HERE / "experiments" / "results"
OLD = RES

def w(name, lines):
    open(T / name, "w", encoding="utf-8").write("\n".join(lines) + "\n")

names = {'ordered_logit': 'Ordered logit', 'multinomial_lr': 'Multinomial logit', 'lc_logit': 'Latent-class ordered logit',
         'rp_logit': 'Random-parameters ordered logit', 'histgb': 'Gradient boosting (balanced)', 'dlcon': 'DLCON', 'tabpfn': 'TabPFN v3'}
order = ['ordered_logit', 'multinomial_lr', 'lc_logit', 'rp_logit', 'histgb', 'dlcon', 'tabpfn']

# ---- Table: seven models (pooled coverage, pooled width, certified width, RPS, timings)
e1 = pd.read_parquet(OLD / "e1_marginal.parquet"); e1 = e1[e1.alpha == 0.10]
mon = pd.read_parquet(OLD / "adhoc_r4_e3_mondrian.parquet"); mon = mon[mon.stratum == 'ALL']
q = json.load(open(RES / "q1_quality.json"))
def tfmt(s):
    s = float(s)
    if s < 1: return r"$<$1\,s"
    if s < 60: return f"{s:.0f}\\,s"
    return f"{s/60:.0f}\\,min"
L = [r"\begin{table}[pos=htbp]",
     r"\caption{One certification layer over seven base models ($\alpha=0.10$).}\label{tbl:models}",
     r"\centering\footnotesize", r"\begin{threeparttable}",
     r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lcccccccc@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Pooled calibration} & Certified & & & & & \\",
     r"\cmidrule(lr){2-3}",
     r"Base model & Coverage & Width & width & RPS & Log loss & Fit & Predict & Calibrate \\", r"\midrule"]
for m in order:
    r = e1[e1.model == m].iloc[0]; c = mon[mon.model == m].iloc[0]
    L.append(f"{names[m]} & {r.coverage:.4f} & {r.avg_width:.2f} & {c.avg_width:.2f} & {q[m]['rps']:.3f} & {q[m]['log_loss']:.3f} & {tfmt(r.fit_s)} & {tfmt(r.predict_s)} & {r.calibrate_s:.2f}\\,s \\\\")
L += [r"\bottomrule", r"\end{tabular*}", r"\begin{tablenotes}[flushleft]\footnotesize",
      r"\item[] \textit{Note:} Held-out test records $n=810{,}082$; nominal coverage 0.90. Pooled calibration uses one threshold for all records; the certified width uses calibration within the four final cells (Table~\ref{tab:maincells}). RPS is the ranked probability score and log loss the mean negative log-likelihood of the reported category on the test fold (lower is better). Every set is contiguous. At $\alpha\in\{0.05,0.10,0.20\}$, 20 of 21 model-by-$\alpha$ cells met a descriptive screen of nominal less three binomial standard errors; the exception is gradient boosting at $\alpha=0.10$, short by 0.00008. Fit and predict belong to the base model and reflect the training subsamples and context sizes in Section~\ref{sec:splitting}; five models ran on a desktop CPU and DLCON and TabPFN on a GPU, so timings compare orders of magnitude only.",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
w("table3_models.tex", L)

# ---- Table: alternative conformal methods
b = pd.read_parquet(RES / "b1_baselines.parquet")
L = [r"\begin{table}[pos=htbp]",
     r"\caption{Alternative conformal methods on the primary sample ($\alpha=0.10$).}\label{tab:baselines}",
     r"\centering\footnotesize", r"\begin{threeparttable}",
     r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}llcccccc@{}}", r"\toprule",
     r" & & & & & \multicolumn{3}{c}{Coverage in safety cells} \\", r"\cmidrule(lr){6-8}",
     r"Base model & Method & Coverage & Mean size & Contiguous & Motorcycle & Rural high-speed & Unrestrained \\", r"\midrule"]
mlab = {'LAC marginal': 'LAC, pooled', 'Label-conditional LAC': 'LAC, label-conditional', 'APS': 'APS',
        'CHOIR marginal': 'CHOIR, pooled', 'CHOIR four-cell': 'CHOIR, four cells'}
for m in ['ordered_logit', 'dlcon', 'histgb']:
    for i, r in b[b.model == m].reset_index().iterrows():
        mn = names[m] if i == 0 else ""
        L.append(f"{mn} & {mlab[r.method]} & {r.coverage:.3f} & {r.mean_size:.2f} & {100*r.contiguous_share:.1f}\\% & {r.cov_motorcycle:.3f} & {r.cov_rural_highspeed:.3f} & {r.cov_unrestrained:.3f} \\\\")
    if m != 'histgb': L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular*}", r"\begin{tablenotes}[flushleft]\footnotesize",
      r"\item[] \textit{Note:} All methods use the same cached base-model probabilities, calibration fold and test fold ($n=810{,}082$). LAC uses the score $1-\hat p(\yt\mid x)$ with one threshold (pooled) or one threshold per KABCO category (label-conditional, the class-conditional construction of prior injury-risk work). APS is the non-randomized adaptive prediction set. CHOIR uses the ordinal cumulative score with one threshold (pooled) or within the four final cells. Contiguous is the share of sets that form one run of adjacent KABCO categories.",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
w("table_baselines.tex", L)

# ---- Table: temporal (corrected split)
e5 = pd.read_parquet(RES / "e5_temporal_split.parquet")
g = lambda meth, yr: e5[(e5.method == meth) & (e5.year == yr)].iloc[0]
L = [r"\begin{table}[pos=htbp]",
     r"\caption{Temporal completed-record stress test and covariate-mismatch statistic ($\alpha=0.10$).}",
     r"\label{tbl:temporal}", r"\centering\footnotesize", r"\begin{threeparttable}",
     r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
     r"Method & Cov. 2024 & Cov. 2025 & Width 2024 & Width 2025 & Mismatch statistic \\", r"\midrule"]
for meth, lab in [('unweighted', 'Unweighted split conformal'), ('county_mondrian_4a', 'County-Mondrian'), ('density_ratio_4b', 'Density-ratio weighted')]:
    a, c = g(meth, 2024), g(meth, 2025)
    ms = f"{a.tv_lcb:.3f} / {c.tv_lcb:.3f}" if meth == 'density_ratio_4b' else "--"
    L.append(f"{lab} & {a.coverage:.4f} & {c.coverage:.4f} & {a.avg_width:.3f} & {c.avg_width:.3f} & {ms} \\\\")
a, c = g('density_ratio_4b', 2024), g('density_ratio_4b', 2025)
L += [r"\bottomrule", r"\end{tabular*}", r"\begin{tablenotes}[flushleft]\footnotesize",
      r"\item[] \textit{Note:} County-Mondrian calibrates within each county that has at least 1,000 training records and pools the rest into one statewide cell. The unweighted and County-Mondrian rows use 593,685 records in 2024 and 577,726 in 2025. For the density-ratio row, the 2022--2023 calibration fold (1,184,540 records) is split at random into a reference half, used with a target-year fit half to estimate $dP_{\mathrm{target},X}/dP_{\mathrm{calibration},X}$, and a scored half (592,270 records) whose scores are weighted. Coverage is evaluated on the other target-year half " + f"({int(a.n):,} and {int(c.n):,} records); same-row unweighted coverage is {a.unw_same_rows_coverage:.4f} and {c.unw_same_rows_coverage:.4f}. The mismatch statistic is the classifier lower bound of Theorem~\\ref{{thm:transfer}}(ii); it is not a bound on the population transfer penalty.",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
w("table5_temporal.tex", L)

# ---- Table: composition, HGB and DLCON, category-dependent map
def comp(m):
    f = RES / ("e8_composition.parquet" if m == 'histgb' else f"e8_composition_{m}.parquet")
    d = pd.read_parquet(f); return d[d.band == 'kabco_catdep'].set_index('final_calibration_cell')
H, D = comp('histgb'), comp('dlcon')
lab = {'cell:baseline|rural': 'Baseline, rural', 'cell:baseline|urban': 'Baseline, urban', 'cell:motorcycle|rural': 'Motorcycle, rural',
       'cell:motorcycle|urban': 'Motorcycle, urban', 'cell:rural_highspeed|rural': 'Rural high-speed, rural',
       'cell:unrestrained|rural': 'Unrestrained, rural', 'cell:unrestrained|urban': 'Unrestrained, urban', 'overall:ALL': 'Overall'}
L = [r"\begin{table}[pos=htbp]",
     r"\caption{Semi-synthetic composition audit by final calibration cell for a wide and a narrow base model ($\alpha=0.10$, $\delta=0.02$).}",
     r"\label{tbl:compose}", r"\centering\footnotesize", r"\begin{threeparttable}",
     r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}", r"\toprule",
     r" & & \multicolumn{3}{c}{Gradient boosting (balanced)} & \multicolumn{3}{c}{DLCON} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"Final calibration cell & $n_{\rm cal}$ & Coverage & Width & Full (\%) & Coverage & Width & Full (\%) \\", r"\midrule"]
for k, v in lab.items():
    h, d = H.loc[k], D.loc[k]
    L.append(f"{v} & {int(h.final_n_cal):,} & {h.coverage:.4f} & {h.mean_width:.2f} & {100*h.frac_full_scale:.1f} & {d.coverage:.4f} & {d.mean_width:.2f} & {100*d.frac_full_scale:.1f} \\\\")
L += [r"\bottomrule", r"\end{tabular*}", r"\begin{tablenotes}[flushleft]\footnotesize",
      r"\item[] \textit{Note:} Rows are the seven non-empty cells of the product of the four declared strata and the rural or urban indicator; no cell required rollup. Sets are expanded with the category-dependent map of Remark~\ref{rem:catdep}; the constant one-category band gives the same values to four decimals. Observed CRIS KABCO is the proxy outcome; synthetic reported labels are generated on calibration rows only, and test coverage is evaluated against test-fold KABCO. Full (\%) is the share of records whose set spans all five categories. Every cell meets the declared 0.88 floor.",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
w("table6_compose.tex", L)

# ---- Table: endpoint audit for three models
L = [r"\begin{table}[pos=htbp]", r"\caption{Completed-record endpoint audit on 810,082 test records.}", r"\label{tab:endpoint}",
     r"\centering\footnotesize", r"\begin{threeparttable}",
     r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}llrrrrr@{}}", r"\toprule",
     r"Base model & Interval & Mean width & Workload & Severe omissions & Conditional omission & Joint omission \\", r"\midrule"]
for m, suf in [('histgb', ''), ('ordered_logit', '_ordered_logit'), ('dlcon', '_dlcon')]:
    d = pd.read_parquet(RES / f"e7_endpoint_audit{suf}.parquet"); d = d[d.final_cell == 'overall']
    for i, (st, sl) in enumerate([('raw_reported_interval', 'Reported-label'), ('expanded_sensitivity_interval', 'Expanded sensitivity')]):
        r = d[d.set_type == st].iloc[0]
        L.append(f"{names[m] if i == 0 else ''} & {sl} & {r.avg_width:.2f} & {r.workload:.4f} & {int(r.n_severe_omitted):,} & {r.conditional_severe_omission:.4f} & {r.joint_severe_omission:.5f} \\\\")
L += [r"\bottomrule", r"\end{tabular*}", r"\begin{tablenotes}[flushleft]\footnotesize",
      r"\item[] \textit{Note:} Intervals use calibration within the four final cells and the observed police-reported KABCO outcome. A record enters the retrospective review queue when the interval upper endpoint is A or K. Workload and joint omission use all test records; conditional omission uses the 13,641 observed A or K test outcomes.",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
w("table_endpoint.tex", L)

# ---- Semi-synthetic per-cell table: Note style
p = T / "table_semi_synthetic.tex"; s = open(p, encoding="utf-8").read()
s = s.replace(r"\begin{tablenotes}\footnotesize" + "\n" + r"\item The fixed-score", r"\begin{tablenotes}[flushleft]\footnotesize" + "\n" + r"\item[] \textit{Note:} The fixed-score")
s = s.replace(r"\caption{Semi-synthetic audit under the declared compatibility relation.}", r"\caption{Semi-synthetic sensitivity audit by final cell under the declared compatibility relation ($\delta_a=0.02$).}")
open(p, "w", encoding="utf-8").write(s)
print("tables written")
