"""Rebuild the seven-model final-cell table (Table 5) from the stored result files."""
import pandas as pd
from pathlib import Path
from scipy.stats import beta

RES = Path(__file__).resolve().parent / "experiments" / "results"
OUT = Path(__file__).resolve().parent / "tables" / "table_main_cells.tex"
m = pd.read_parquet(RES / "adhoc_r4_e3_mondrian.parquet")
g = pd.read_parquet(RES / "adhoc_r4_e2_marginal.parquet")
names = {'ordered_logit': 'Ordered logit', 'multinomial_lr': 'Multinomial logit', 'lc_logit': 'Latent-class OL',
         'rp_logit': 'Random-parameters OL', 'histgb': 'Gradient boosting', 'dlcon': 'DLCON', 'tabpfn': 'TabPFN'}
order = ['ordered_logit', 'multinomial_lr', 'lc_logit', 'rp_logit', 'histgb', 'dlcon', 'tabpfn']
strata = ['baseline', 'motorcycle', 'rural_highspeed', 'unrestrained']
a = 0.05 / 28


def ci(c, n):
    k = int(round(c * n)); return beta.ppf(a / 2, k, n - k + 1), beta.ppf(1 - a / 2, k + 1, n - k)


def row(df, md, st):
    return df[(df.model == md) & (df.stratum == st)].iloc[0]


L = [r"\begin{table}[pos=htbp]",
     r"\caption{Coverage and width by final cell for all seven base models ($\alpha=0.10$).}",
     r"\label{tab:maincells}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{threeparttable}",
     r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lccccc@{}}", r"\toprule",
     r"Base model & Baseline & Motorcycle & Rural high-speed & Unrestrained & All \\", r"\midrule",
     r"\multicolumn{6}{@{}l}{\textsc{Panel A. Final-cell (certified) coverage, with simultaneous 95\% interval}}\\", r"\midrule"]
for md in order:
    cells = []
    for st in strata:
        r = row(m, md, st); lo, hi = ci(r.coverage, int(r.n))
        cells.append(f"{r.coverage:.4f} [{lo:.3f}, {hi:.3f}]")
    L.append(names[md] + " & " + " & ".join(cells) + f" & {row(m, md, 'ALL').coverage:.4f} \\\\")
L += [r"\midrule", r"\multicolumn{6}{@{}l}{\textsc{Panel B. Final-cell (certified) mean set width, KABCO categories}}\\", r"\midrule"]
for md in order:
    L.append(names[md] + " & " + " & ".join(f"{row(m, md, st).avg_width:.2f}" for st in strata + ['ALL']) + r" \\")
L += [r"\midrule", r"\multicolumn{6}{@{}l}{\textsc{Panel C. Coverage under pooled (marginal) calibration}}\\", r"\midrule"]
for md in order:
    L.append(names[md] + " & " + " & ".join(f"{row(g, md, st).coverage:.4f}" for st in strata + ['ALL']) + r" \\")
L += [r"\midrule", r"Test records $n$ & 699,412 & 8,530 & 91,805 & 10,335 & 810,082 \\", r"\bottomrule", r"\end{tabular*}",
      r"\begin{tablenotes}[flushleft]\footnotesize",
      r"\item[] \textit{Note:} The nominal coverage is 0.90. The 28 two-sided Clopper--Pearson intervals in Panel A use a Bonferroni correction for at least 95\% simultaneous coverage, and every interval contains 0.90. The intervals are conditional on the fitted calibration thresholds; the variation across 200 repeated calibration splits is reported in Section~\ref{sec:e1}. Cells are mutually exclusive and follow the priority motorcycle, unrestrained among the remainder, rural high-speed among the remainder, and baseline. Panel C uses one pooled threshold for all records; its binomial standard errors are about 0.004 in the motorcycle and unrestrained cells. OL is ordered logit; gradient boosting uses balanced class weights.",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("table 5 written")
