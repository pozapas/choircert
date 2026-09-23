#!/usr/bin/env python
"""Generate focused result tables from stored aggregate result files."""

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "ordered_logit": "Ordered logit",
    "histgb": "Histogram gradient boosting",
    "dlcon": "DLCON",
}
CELL_NAMES = {
    "baseline": "Baseline",
    "motorcycle": "Motorcycle",
    "rural_highspeed": "Rural high-speed",
    "unrestrained": "Unrestrained",
    "overall": "Overall",
}


def write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_cells():
    data = (
        pl.read_parquet(RESULTS / "e2_main_cells.parquet")
        .filter(pl.col("method") == "final_cell")
        .sort(["model", "final_cell"])
        .to_dicts()
    )
    order = {m: i for i, m in enumerate(MODEL_NAMES)}
    cell_order = {c: i for i, c in enumerate(CELL_NAMES)}
    data.sort(key=lambda r: (order[r["model"]], cell_order[r["final_cell"]]))
    lines = [
        r"\begin{table}[pos=htbp]",
        r"\caption{Reported-label coverage within the four training-frozen final cells.}",
        r"\label{tab:maincells}",
        r"\centering\footnotesize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Final cell & $n$ & Coverage & Simultaneous 95\% interval & Mean width \\",
        r"\midrule",
    ]
    current = None
    for row in data:
        if current is not None and row["model"] != current:
            lines.append(r"\addlinespace")
        current = row["model"]
        lines.append(
            f'{MODEL_NAMES[row["model"]]} & {CELL_NAMES[row["final_cell"]]} & '
            f'{row["n"]:,} & {row["coverage"]:.4f} & '
            f'[{row["simultaneous_ci_low"]:.4f}, {row["simultaneous_ci_high"]:.4f}] & '
            f'{row["avg_width"]:.2f} \\\\'
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\footnotesize",
        r"\item The nominal coverage is 0.90. The 12 two-sided Clopper--Pearson intervals use Bonferroni correction for at least 95\% simultaneous coverage. Cells are mutually exclusive and follow the priority motorcycle, unrestrained among the remainder, rural high-speed among the remainder, and baseline.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    write(TABLES / "table_main_cells.tex", lines)


def temporal_cells():
    data = (
        pl.read_parquet(RESULTS / "e6_temporal_cells.parquet")
        .sort(["year", "final_cell"])
        .to_dicts()
    )
    cell_order = {c: i for i, c in enumerate(CELL_NAMES)}
    data.sort(key=lambda r: (r["year"], cell_order[r["final_cell"]]))
    lines = [
        r"\begin{table}[pos=htbp]",
        r"\caption{Final-cell temporal stress test for histogram gradient boosting.}",
        r"\label{tab:temporalcells}",
        r"\centering\footnotesize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Year & Final cell & $n$ & Coverage & Simultaneous 95\% interval & Mean width \\",
        r"\midrule",
    ]
    current = None
    for row in data:
        if current is not None and row["year"] != current:
            lines.append(r"\addlinespace")
        current = row["year"]
        lines.append(
            f'{row["year"]} & {CELL_NAMES[row["final_cell"]]} & {row["n"]:,} & '
            f'{row["coverage"]:.4f} & '
            f'[{row["simultaneous_ci_low"]:.4f}, {row["simultaneous_ci_high"]:.4f}] & '
            f'{row["avg_width"]:.2f} \\\\'
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\footnotesize",
        r"\item The model is trained on 2017--2021 records and calibrated on 2022--2023 records. The eight two-sided Clopper--Pearson intervals use Bonferroni correction for at least 95\% simultaneous coverage. This table is a descriptive temporal stress test.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    write(TABLES / "table_temporal_cells.tex", lines)


def semi_synthetic():
    data = (
        pl.read_parquet(RESULTS / "e4_noise.parquet")
        .filter((pl.col("band") == "kabco_main") & (pl.col("delta") == 0.02))
        .to_dicts()
    )
    cell_order = {c: i for i, c in enumerate(CELL_NAMES)}
    data.sort(key=lambda r: cell_order[r["final_cell"]])
    lines = [
        r"\begin{table}[pos=htbp]",
        r"\caption{Semi-synthetic audit under the declared compatibility relation.}",
        r"\label{tab:semisynthetic}",
        r"\centering\footnotesize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Final cell & $n$ & Report coverage & Outside-map share & Expanded coverage & Mean width \\",
        r"\midrule",
    ]
    for row in data:
        lines.append(
            f'{CELL_NAMES[row["final_cell"]]} & {row["n"]:,} & '
            f'{row["reported_coverage"]:.4f} & {row["beyond_map_test"]:.4f} & '
            f'{row["coverage"]:.4f} & {row["avg_width"]:.2f} \\\\'
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\footnotesize",
        r"\item The fixed-score implementation stress test uses $\alpha=0.10$, $\delta_a=0.02$, and seed 20260708. The theorem floor is 0.88 in every cell. The observed CRIS category is the pre-injection proxy label. Cells use the priority rule stated in Table~\ref{tab:maincells}.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    write(TABLES / "table_semi_synthetic.tex", lines)


def endpoint():
    data = (
        pl.read_parquet(RESULTS / "e7_endpoint_audit.parquet")
        .filter(pl.col("final_cell") == "overall")
        .to_dicts()
    )
    names = {
        "raw_reported_interval": "Reported-label interval",
        "expanded_sensitivity_interval": "Expanded sensitivity interval",
    }
    lines = [
        r"\begin{table}[pos=htbp]",
        r"\caption{Completed-record endpoint audit on 810,082 test records.}",
        r"\label{tab:endpoint}",
        r"\centering\footnotesize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Interval & Mean width & Workload & Severe omissions & Conditional omission & Joint omission \\",
        r"\midrule",
    ]
    for row in data:
        lines.append(
            f'{names[row["set_type"]]} & {row["avg_width"]:.2f} & '
            f'{row["workload"]:.4f} & {row["n_severe_omitted"]:,} & '
            f'{row["conditional_severe_omission"]:.4f} & '
            f'{row["joint_severe_omission"]:.5f} \\\\'
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\footnotesize",
        r"\item The intervals use histogram gradient boosting and the original police-reported KABCO outcome. A record enters the retrospective review queue when the interval upper endpoint is A or K. Workload and joint omission use all test records. Conditional omission uses the 13,641 observed A or K test outcomes.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    write(TABLES / "table_endpoint.tex", lines)


def temporal_transfer():
    data = pl.read_parquet(RESULTS / "e5_temporal.parquet").to_dicts()
    method_order = ["unweighted", "county_mondrian_4a", "density_ratio_4b"]
    names = {
        "unweighted": "Unweighted split conformal",
        "county_mondrian_4a": "County-Mondrian",
        "density_ratio_4b": "Density-ratio weighted",
    }
    lookup = {(row["method"], row["year"]): row for row in data}
    lines = [
        r"\begin{table}[pos=htbp]",
        r"\caption{Temporal completed-record stress test and covariate-mismatch diagnostic (E5, $\alpha=0.10$).}",
        r"\label{tbl:temporal}",
        r"\centering\footnotesize",
        r"\begin{threeparttable}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
        r"\toprule",
        r"Method & Cov. 2024 & Cov. 2025 & Width 2024 & Width 2025 & Empirical LCB \\",
        r"\midrule",
    ]
    for method in method_order:
        a, b = lookup[(method, 2024)], lookup[(method, 2025)]
        tv = ("--" if method != "density_ratio_4b"
              else f'{a["tv_lcb"]:.3f} / {b["tv_lcb"]:.3f}')
        lines.append(
            f'{names[method]} & {a["coverage"]:.4f} & {b["coverage"]:.4f} & '
            f'{a["avg_width"]:.3f} & {b["avg_width"]:.3f} & {tv} \\\\'
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\begin{tablenotes}\footnotesize",
        r"\item The unweighted and County-Mondrian rows use 593,685 records in 2024 and 577,726 in 2025. The density-ratio row evaluates 296,843 and 288,863 records after estimating $dP_{\mathrm{target},X}/dP_{\mathrm{calibration},X}$ from 1,184,540 calibration records and disjoint target-year fit halves. Same-row unweighted coverage is 0.8990 and 0.9006. The empirical lower confidence bound measures separation from the finite weighted reference and does not bound the population transfer penalty. Both rerun values are 0.000.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    write(TABLES / "table5_temporal.tex", lines)


def composition():
    data = (
        pl.read_parquet(RESULTS / "e8_composition.parquet")
        .filter(pl.col("band") == "constant_b1")
        .to_dicts()
    )
    def sort_key(row):
        return (row["rollup_level"] == "overall", row["final_calibration_cell"])
    data.sort(key=sort_key)
    lines = [
        r"\begin{table}[pos=htbp]",
        r"\caption{Semi-synthetic composition audit by resolved final calibration cell (E8, $\alpha=0.10$, $\delta=0.02$).}",
        r"\label{tbl:compose}",
        r"\centering\footnotesize",
        r"\begin{threeparttable}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
        r"\toprule",
        r"Final calibration cell & $n_{\rm cal}$ & $n_{\rm test}$ & Coverage & Mean width & Full scale (\%) \\",
        r"\midrule",
    ]
    for row in data:
        label = ("Overall" if row["rollup_level"] == "overall" else
                 CELL_NAMES[row["final_calibration_cell"].split(":")[1].split("|")[0]]
                 + ", " + row["final_calibration_cell"].split("|")[1])
        lines.append(
            f'{label} & {row["final_n_cal"]:,} & {row["n"]:,} & '
            f'{row["coverage"]:.4f} & {row["avg_width"]:.3f} & '
            f'{100 * row["frac_full_scale"]:.1f} \\\\'
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\begin{tablenotes}\footnotesize",
        r"\item Observed CRIS KABCO is the proxy outcome on calibration and test folds. Synthetic reported labels are generated on calibration rows only; test coverage is evaluated against test-fold KABCO. Rows are resolved final calibration cells after the declared product-cell rollup. All cells meet the declared 0.88 sensitivity floor. The table reports a semi-synthetic implementation audit against proxy KABCO.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    write(TABLES / "table6_compose.tex", lines)


if __name__ == "__main__":
    main_cells()
    temporal_cells()
    semi_synthetic()
    endpoint()
    temporal_transfer()
    composition()
    print("Wrote result tables")
