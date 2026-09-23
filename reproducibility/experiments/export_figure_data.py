#!/usr/bin/env python
"""Export thin CSV tables for the CHOIR TikZ/PGFPlots figure suite.

The experiment results remain the authoritative files. This script only extracts
the small plotting tables that the standalone LaTeX figures consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "figure_src" / "data"
OUT.mkdir(parents=True, exist_ok=True)
ANALYSIS = ROOT / "data" / "processed" / "analysis.parquet"


def read_analysis(columns: list[str]) -> pd.DataFrame:
    """Read the primary-row snapshot with a Windows-safe fastparquet fallback."""

    try:
        df = pd.read_parquet(ANALYSIS, columns=columns + ["primary_row"])
    except Exception:
        df = pd.read_parquet(ANALYSIS, columns=columns + ["primary_row"], engine="fastparquet")
    df = df[df["primary_row"].astype(bool)].drop(columns=["primary_row"])
    return df.reset_index(drop=True)


def read_result(name: str) -> pd.DataFrame:
    """Read a result parquet with a Windows-safe fallback.

    Some synced parquet files in this workspace fail with pyarrow 19 on Windows
    but read cleanly with fastparquet. Keeping the fallback here avoids changing
    the frozen experiment outputs.
    """

    path = RESULTS / name
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, float_format="%.6f", na_rep="nan")


def export_fig02() -> None:
    df = read_result("e2_e3_heterogeneity.parquet")
    order = ["baseline", "rural_highspeed", "motorcycle", "unrestrained"]
    y_pos = {cell: i + 1 for i, cell in enumerate(order)}

    marginal = (
        df[(df.exp == "E2") & (df.cell_type == "declared")]
        .set_index("cell")
        .loc[order]
        .reset_index()
    )
    mondrian = (
        df[(df.method == "mondrian_declared") & (df.cell_type == "declared")]
        .set_index("cell")
        .loc[order]
        .reset_index()
    )
    declared = pd.DataFrame(
        {
            "cell": order,
            "y": [y_pos[c] for c in order],
            "marginal": marginal["coverage"].to_numpy(),
            "mondrian": mondrian["coverage"].to_numpy(),
            "width_marginal": marginal["avg_width"].to_numpy(),
            "width_mondrian": mondrian["avg_width"].to_numpy(),
            "n": marginal["n"].to_numpy(),
        }
    )
    write_csv(declared, "fig02_declared.csv")
    links = []
    for row in declared.itertuples(index=False):
        links.append({"x": row.marginal, "y": row.y})
        links.append({"x": row.mondrian, "y": row.y})
        links.append({"x": float("nan"), "y": float("nan")})
    write_csv(pd.DataFrame(links), "fig02_links.csv")

    kabco_order = ["O", "C", "B", "A", "K"]
    kabco = (
        df[(df.exp == "E2") & (df.cell_type == "kabco")]
        .set_index("cell")
        .loc[kabco_order]
        .reset_index()
    )
    kabco["x"] = range(1, len(kabco) + 1)
    kabco["shortfall"] = (0.9 - kabco["coverage"]).clip(lower=0)
    write_csv(kabco[["cell", "x", "coverage", "avg_width", "n", "shortfall"]], "fig02_kabco.csv")


def export_fig03() -> None:
    df = read_result("e2_e3_heterogeneity.parquet")
    rows = []
    labels = {
        "marginal": "Marginal",
        "mondrian_lc_logit_gate": "LC logit",
        "mondrian_declared": "Declared",
        "mondrian_dlcon_gate": "DLCON gate",
        "mondrian_kmeans8": "KMeans-8",
    }
    for method, label in labels.items():
        if method == "marginal":
            row = df[(df.exp == "E2") & (df.method == method) & (df.cell_type == "overall")].iloc[0]
        else:
            row = df[(df.exp == "E3") & (df.method == method) & (df.cell_type == "overall")].iloc[0]
        rows.append({"method": method, "label": label, "width": row.avg_width, "coverage": row.coverage})
    part = pd.DataFrame(rows)
    part["x"] = range(1, len(part) + 1)
    write_csv(part[["method", "label", "x", "width", "coverage"]], "fig03_partitions.csv")

    lc = df[(df.method == "mondrian_lc_logit_gate") & (df.cell_type == "lc_logit_gate")].copy()
    lc["x"] = range(1, len(lc) + 1)
    write_csv(lc[["cell", "x", "coverage", "avg_width", "n"]], "fig03_lc_classes.csv")


def export_fig02_landscape() -> None:
    """Fig 2 (new slate, §4): severity landscape over (hour x speed-limit)."""

    df = read_analysis(["speed_limit", "hour", "y_kabco"])
    df = df[df["speed_limit"] > 0].copy()
    bins = np.arange(15, 90, 5)
    labels = bins[:-1]
    df["speed_bin"] = pd.cut(df["speed_limit"], bins=bins, labels=labels, right=False)
    df = df.dropna(subset=["speed_bin"])
    df["speed_bin"] = df["speed_bin"].astype(int)
    df["is_ka"] = df["y_kabco"].isin([4, 5])

    grid = (
        df.groupby(["hour", "speed_bin"])
        .agg(n=("is_ka", "size"), n_ka=("is_ka", "sum"))
        .reset_index()
    )
    full_index = pd.MultiIndex.from_product(
        [range(24), labels.tolist()], names=["hour", "speed_bin"]
    )
    grid = grid.set_index(["hour", "speed_bin"]).reindex(full_index, fill_value=0).reset_index()
    grid["share_ka"] = np.where(grid["n"] > 0, grid["n_ka"] / grid["n"], np.nan)
    write_csv(grid[["hour", "speed_bin", "n", "n_ka", "share_ka"]], "fig02_landscape.csv")


def export_fig03_corner() -> None:
    """Fig 3 (new slate, §5): covariate geometry by outcome (O vs K+A corner)."""

    cols = ["speed_limit", "hour", "age", "vehicle_age", "y_kabco"]
    df = read_analysis(cols)
    df = df[df["speed_limit"] > 0].dropna(subset=cols)

    group_o = df[df["y_kabco"] == 1]
    group_ka = df[df["y_kabco"].isin([4, 5])]

    meta = {
        "n_o": int(len(group_o)),
        "n_ka": int(len(group_ka)),
        "seed": 20260704,
        "n_sample": 150_000,
        "variables": ["speed_limit", "hour", "age", "vehicle_age"],
    }
    for name, group in [("o", group_o), ("ka", group_ka)]:
        n_sample = min(meta["n_sample"], len(group))
        sample = group.sample(n=n_sample, random_state=meta["seed"])[
            ["speed_limit", "hour", "age", "vehicle_age"]
        ]
        write_csv(sample, f"fig03_corner_{name}.csv")
        meta[f"median_{name}"] = {c: float(sample[c].median()) for c in sample.columns}
        meta[f"iqr_{name}"] = {
            c: [float(sample[c].quantile(0.25)), float(sample[c].quantile(0.75))]
            for c in sample.columns
        }

    import json

    (OUT / "fig03_corner_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def export_fig04() -> None:
    df = read_result("e4_noise.parquet")
    keep = df[df.band.isin(["constant_b1", "kabco_catdep"])].copy()
    write_csv(
        keep[["band", "delta", "eps_tot", "floor", "floor_unstructured", "coverage", "avg_width"]],
        "fig04_noise.csv",
    )
    write_csv(
        keep[keep.band == "kabco_catdep"][
            ["delta", "eps_tot", "floor", "floor_unstructured", "coverage", "avg_width"]
        ].sort_values("delta"),
        "fig04_catdep.csv",
    )
    write_csv(
        keep[keep.band == "constant_b1"][
            ["delta", "eps_tot", "floor", "floor_unstructured", "coverage", "avg_width"]
        ].sort_values("delta"),
        "fig04_constant.csv",
    )


def export_fig05() -> dict[str, float]:
    temporal = read_result("e5_temporal.parquet")
    if "best_case_floor_upper_from_tv_lcb" not in temporal.columns:
        raise RuntimeError(
            "e5_temporal.parquet predates the target-to-calibration density-ratio "
            "repair. Rerun e5_temporal.py before exporting Figure 5 data.")
    write_csv(
        temporal[
            [
                "method",
                "year",
                "coverage",
                "avg_width",
                "se",
                "tv_lcb",
                "best_case_floor_upper_from_tv_lcb",
            ]
        ],
        "fig05_temporal.csv",
    )
    for method, filename in [
        ("unweighted", "fig05_temporal_unweighted.csv"),
        ("county_mondrian_4a", "fig05_temporal_county.csv"),
        ("density_ratio_4b", "fig05_temporal_density.csv"),
    ]:
        write_csv(
            temporal[temporal.method == method][
                ["year", "coverage", "avg_width", "se", "tv_lcb",
                 "best_case_floor_upper_from_tv_lcb"]
            ].sort_values("year"),
            filename,
        )
    spatial = read_result("e6_spatial.parquet")
    cert = spatial[spatial.n_test >= 2000]
    summary = {
        "county_min": float(spatial.unw_coverage.min()),
        "county_max": float(spatial.unw_coverage.max()),
        "county_n": int(len(spatial)),
        "county_certified": int(len(cert)),
        "cert_floor_min": float(cert.cert_floor_lcb.min()),
        "cert_floor_max": float(cert.cert_floor_lcb.max()),
    }
    return summary


def export_fig06() -> dict[str, float]:
    risk = read_result("e7_risk.parquet")
    required = {"joint_omission_numerator", "joint_omission_denominator",
                "conditional_fatal_omission"}
    if not required.issubset(risk.columns):
        raise RuntimeError(
            "e7_risk.parquet predates the joint-versus-conditional fatal-omission "
            "repair. Rerun e7_risk.py before exporting Figure 6 data.")
    write_csv(risk[["guarantee", "beta", "risk", "bound", "holds", "n", "n_fatal",
                    "joint_omission_numerator", "joint_omission_denominator",
                    "conditional_fatal_omission"]], "fig06_risk.csv")
    write_csv(
        risk[risk.guarantee == "cost_risk"][["beta", "risk", "bound", "holds"]],
        "fig06_cost.csv",
    )
    write_csv(
        risk[risk.guarantee == "fatal_omission"][[
            "beta", "risk", "bound", "holds", "joint_omission_numerator",
            "joint_omission_denominator", "conditional_fatal_omission", "n_fatal",
        ]],
        "fig06_fatal.csv",
    )
    fatal = risk[risk.guarantee == "fatal_omission"].iloc[0]
    cost = risk[risk.guarantee == "cost_risk"].iloc[0]
    return {
        "joint_fatal_omission": float(fatal.risk),
        "conditional_fatal_omission": float(fatal.conditional_fatal_omission),
        "n_fatal": int(fatal.n_fatal),
        "cost_risk": float(cost.risk),
    }


def export_fig07() -> dict[str, float]:
    e1 = read_result("e1_marginal.parquet")
    models = e1[e1.alpha == 0.10].copy()
    label_order = [
        "dlcon",
        "multinomial_lr",
        "ordered_logit",
        "lc_logit",
        "rp_logit",
        "tabpfn",
        "histgb",
    ]
    label_map = {
        "dlcon": "DLCON",
        "multinomial_lr": "MNL",
        "ordered_logit": "Ordered logit",
        "lc_logit": "LC logit",
        "rp_logit": "RP logit",
        "tabpfn": "TabPFN",
        "histgb": "HistGB",
    }
    models = models.set_index("model").loc[label_order].reset_index()
    models["x"] = range(1, len(models) + 1)
    models["label"] = models["model"].map(label_map)
    write_csv(models[["model", "label", "x", "coverage", "avg_width", "fit_s", "predict_s", "calibrate_s"]], "fig07_models.csv")

    scaling = read_result("e9_layer_scaling.parquet")
    scaling = scaling.copy()
    scaling["n_million"] = scaling["n"] / 1_000_000
    write_csv(scaling[["n", "n_million", "layer_s"]], "fig07_scaling.csv")

    comp = read_result("e8_composition.parquet")
    comp = comp[comp["class"] != "ALL"].copy()
    comp["x"] = range(1, len(comp) + 1)
    write_csv(comp[["class", "stratum", "x", "coverage", "floor", "n"]], "fig07_composition.csv")
    return {
        "scale_one_m": float(scaling.loc[scaling.n == 1_000_000, "layer_s"].iloc[0]),
        "scale_five_m": float(scaling.loc[scaling.n == 5_000_000, "layer_s"].iloc[0]),
        "composition_min": float(comp.coverage.min()),
        "composition_floor": float(comp.floor.iloc[0]),
    }


def write_numbers(numbers: dict[str, float]) -> None:
    def macro(name: str, value: object) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}\n"

    lines = [
        "% Auto-generated by experiments/export_figure_data.py. Do not edit by hand.\n",
        macro("FigTwoTarget", "0.90"),
        macro("FigFiveCountyMin", f"{numbers['county_min']:.3f}"),
        macro("FigFiveCountyMax", f"{numbers['county_max']:.3f}"),
        macro("FigFiveCountyN", int(numbers["county_n"])),
        macro("FigFiveCountyCertified", int(numbers["county_certified"])),
        macro("FigFiveCertFloorMin", f"{numbers['cert_floor_min']:.3f}"),
        macro("FigFiveCertFloorMax", f"{numbers['cert_floor_max']:.3f}"),
        macro("FigSixFatalRate", f"{numbers['joint_fatal_omission']:.4f}"),
        macro("FigSixFatalN", int(numbers["n_fatal"])),
        macro("FigSixCostRisk", f"{numbers['cost_risk']:.2f}"),
        macro("FigSevenScaleOneM", f"{numbers['scale_one_m']:.2f}"),
        macro("FigSevenScaleFiveM", f"{numbers['scale_five_m']:.2f}"),
        macro("FigSevenCompositionMin", f"{numbers['composition_min']:.3f}"),
        macro("FigSevenCompositionFloor", f"{numbers['composition_floor']:.2f}"),
    ]
    (OUT / "figure_numbers.tex").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    export_fig02()
    export_fig03()
    export_fig02_landscape()
    export_fig03_corner()
    export_fig04()
    numbers: dict[str, float] = {}
    numbers.update(export_fig05())
    numbers.update(export_fig06())
    numbers.update(export_fig07())
    write_numbers(numbers)
    print(f"figure data -> {OUT}")


if __name__ == "__main__":
    main()
