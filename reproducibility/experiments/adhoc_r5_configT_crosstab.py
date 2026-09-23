"""Round 5, Task 1: crosstab the seven config_T fields within the motorcycle and
unrestrained strata, to test the supervisor's prediction (SUPERVISOR_RESPONSE_2 sec 1):

  - Prsn_Airbag_ID, Prsn_Ejct_ID expected >=90% "Not Applicable" on motorcycles
    (inapplicability codes -> zero information for that stratum);
  - FHE_Collsn_ID / Harm_Evnt_ID / Obj_Struck_ID expected to carry the real
    motorcyclist signal;
  - Veh_Damage_Severity1_Id expected weak on motorcycles (rider absorbs energy).

For each field x stratum: value distribution, the "Not Applicable"/"Unknown"/null share
(the inapplicability mass), and a crude signal check -- the spread of P(y_kabco>=4 | field
value) across the field's populated levels, which is how much the field moves the
severe-tail probability. A field whose severe-tail rate is flat across its levels carries
no severity signal on that stratum however well populated it is.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
from common import load_primary  # noqa: E402

# these config_T fields are not in common.META/CAT, so pull them straight from the snapshot
import polars as pl  # noqa: E402
from common import SNAPSHOT  # noqa: E402

T_FIELDS = ["FHE_Collsn_ID", "Harm_Evnt_ID", "Obj_Struck_ID", "Veh_Damage_Severity1_Id",
            "Prsn_Airbag_ID", "Prsn_Ejct_ID", "num_units"]
INAPPLIC = {"Not Applicable", "Unknown", "No Data", "Not Reported", "", "NA", "Missing"}


def main():
    base = (pl.scan_parquet(SNAPSHOT)
            .filter(pl.col("primary_row"))
            .select(["y_kabco", "Prsn_Type_ID", "Prsn_Rest_ID"] + T_FIELDS)
            .collect()
            .to_pandas())
    base["y_kabco"] = base["y_kabco"].astype("int8")

    moto = base["Prsn_Type_ID"] == "Driver Of Motorcycle Type Vehicle"
    unre = base["Prsn_Rest_ID"].astype(str).isin({"None"})
    strata = {"motorcycle": moto.to_numpy(),
              "unrestrained": unre.to_numpy(),
              "ALL": np.ones(len(base), bool)}

    for sname, mask in strata.items():
        sub = base.loc[mask]
        n = len(sub)
        print(f"\n{'='*72}\n=== {sname}: n={n:,}  "
              f"(P(y>=4)={np.mean(sub['y_kabco'] >= 4):.4f}, "
              f"P(y==5)={np.mean(sub['y_kabco'] == 5):.4f}) ===")
        for f in T_FIELDS:
            col = sub[f].astype(str)
            vc = col.value_counts(dropna=False)
            inapplic_mass = col.isin(INAPPLIC).mean() + col.isna().mean()
            # severe-tail rate per level, populated levels only (>=1% of stratum)
            g = sub.assign(_v=col.values)
            g = g.groupby("_v")["y_kabco"]
            rate = g.apply(lambda s: float(np.mean(s >= 4)))
            size = g.size()
            keep = size[size >= max(30, 0.01 * n)].index
            r = rate.loc[keep].sort_values()
            spread = (r.max() - r.min()) if len(r) >= 2 else float("nan")
            print(f"\n  [{f}]  levels={col.nunique()}  "
                  f"inapplicable/unknown/null mass={inapplic_mass:.4f}")
            print(f"    top levels: " + ", ".join(
                f"{str(k)[:22]}={v/n:.3f}" for k, v in vc.head(4).items()))
            print(f"    P(y>=4) spread across {len(keep)} populated levels: {spread:.4f}"
                  + ("" if len(keep) >= 2 else "  (too few populated levels to judge)"))
            if len(r) >= 2:
                lo_k, hi_k = r.index[0], r.index[-1]
                print(f"      lowest  {str(lo_k)[:26]:26s} P(y>=4)={r.iloc[0]:.4f}")
                print(f"      highest {str(hi_k)[:26]:26s} P(y>=4)={r.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
