"""Shared experiment infrastructure: data loading, splits, feature encoding, models.

All splits are by crash_id (Remark 0.1); primary analysis rows only (one driver per
crash, flagged in the snapshot). All randomness flows from SEED.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

SEED = 20260704
ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "processed" / "analysis.parquet"
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Config P feature columns (pipeline/features.json), split by kind.
CONT = ["speed_limit", "adt", "hour", "age", "vehicle_age", "doors",
        "displacement_l", "engine_hp"]
CAT = ["Wthr_Cond_ID", "Light_Cond_ID", "Surf_Cond_ID", "Road_Type_ID", "Road_Algn_ID",
       "Traffic_Cntl_ID", "Intrsct_Relat_ID", "Rural_Fl", "Func_Sys_ID", "Road_Cls_ID",
       "Day_of_Week", "Prsn_Gndr_ID", "Prsn_Rest_ID", "Drvr_Lic_Type_ID",
       "Drvr_Lic_Cls_ID", "Prsn_Type_ID", "BodyClass", "VehicleType", "GVWR",
       "DriveType", "ESC", "CIB", "ForwardCollisionWarning", "LaneDepartureWarning",
       "BlindSpotMon", "AirBagLocCurtain", "AirBagLocSide", "Pop_Group_ID"]
BOOL = ["vin_undecoded", "has_geo"]
META = ["crash_id", "crash_year", "crash_month", "y_kabco", "Cnty_ID", "CBSA"]

K = 5
S1_START_YEAR = 2017
S1_END_YEAR = 2023
S1_PROPORTIONS = {"train": 0.60, "calibration": 0.20, "test": 0.20}


def load_primary() -> pd.DataFrame:
    """Load primary one-driver-per-crash rows with the columns experiments need."""
    cols = list(dict.fromkeys(META + CONT + CAT + BOOL))
    df = (
        pl.scan_parquet(SNAPSHOT)
        .filter(pl.col("primary_row"))
        .select(cols)
        .collect()
        # Keep Arrow-backed columns during transfer. Converting all categorical
        # fields to Python objects at once requires more than one gigabyte for the
        # current snapshot and can fail before the compact category conversion below.
        .to_pandas(use_pyarrow_extension_array=True)
    )
    df["y_kabco"] = df["y_kabco"].astype(np.int8)
    for c in CONT:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    for c in BOOL:
        df[c] = df[c].astype(np.float32)
    for c in CAT:
        df[c] = df[c].fillna("Missing").replace("", "Missing").astype("category")
    return df


# ---------------------------------------------------------------------------
# Splits (crash-level; primary rows are already one per crash)
# ---------------------------------------------------------------------------

def split_s1(df: pd.DataFrame, seed: int = SEED):
    """Random 60/20/20 S1 split within calendar years 2017 through 2023.

    This explicit lower bound prevents a future snapshot with pre-2017 records from
    silently changing the S1 estimand. The split is at the crash level because the
    primary analysis table contains one row per crash.
    """
    d = df[(df.crash_year >= S1_START_YEAR) & (df.crash_year <= S1_END_YEAR)]
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=len(d))
    tr = d[u < 0.6]
    ca = d[(u >= 0.6) & (u < 0.8)]
    te = d[u >= 0.8]
    return tr, ca, te


def s1_split_audit(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Return the reproducible S1 population, year, and fold-count ledger."""
    tr, ca, te = split_s1(df, seed=seed)
    eligible = df[(df.crash_year >= S1_START_YEAR) &
                  (df.crash_year <= S1_END_YEAR)]
    if eligible["crash_id"].duplicated().any():
        raise ValueError("S1 requires one primary row per crash_id")
    return {
        "split": "S1",
        "population": "primary one-driver-per-crash analysis rows",
        "year_start": S1_START_YEAR,
        "year_end": S1_END_YEAR,
        "seed": int(seed),
        "proportions": S1_PROPORTIONS,
        "eligible_rows": int(len(eligible)),
        "eligible_unique_crashes": int(eligible["crash_id"].nunique()),
        "fold_counts": {
            "train": int(len(tr)), "calibration": int(len(ca)), "test": int(len(te)),
        },
        "year_counts": {str(int(k)): int(v) for k, v in
                        eligible["crash_year"].value_counts().sort_index().items()},
    }


def split_s2(df: pd.DataFrame):
    """Temporal: train 2017-2021, calibrate 2022-2023, test 2024-2025."""
    return (df[df.crash_year <= 2021],
            df[(df.crash_year >= 2022) & (df.crash_year <= 2023)],
            df[df.crash_year >= 2024])


def split_s3(df: pd.DataFrame, n_holdout: int = 54, seed: int = SEED):
    """Spatial: hold out counties (stratified by rural share) for testing.

    Train/calibrate on the remaining 200 counties (random 75/25 within), test on
    held-out counties. Returns (train, cal, test, holdout_counties).
    """
    d = df[df.crash_year <= 2023]
    rng = np.random.default_rng(seed + 3)
    counties = d.groupby("Cnty_ID", observed=True)["Rural_Fl"] \
                .apply(lambda s: (s == "Y").mean()).sort_values()
    # stratified systematic sample over the rural-share ordering
    idx = np.linspace(0, len(counties) - 1, n_holdout).round().astype(int)
    holdout = set(counties.index[idx])
    te = d[d.Cnty_ID.isin(holdout)]
    rest = d[~d.Cnty_ID.isin(holdout)]
    u = rng.uniform(size=len(rest))
    return rest[u < 0.75], rest[u >= 0.75], te, sorted(holdout)


# ---------------------------------------------------------------------------
# Feature encoding (shared design matrix for all matrix models)
# ---------------------------------------------------------------------------

def make_encoder():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cont = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                        sparse_output=False, dtype=np.float32)
    return ColumnTransformer(
        [("cont", cont, CONT), ("cat", cat, CAT), ("bool", "passthrough", BOOL)],
        verbose_feature_names_out=False,
    )


def encode(enc, df: pd.DataFrame, chunk: int = 500_000) -> np.ndarray:
    """Transform in chunks to bound peak memory; float32 output."""
    parts = []
    for i in range(0, len(df), chunk):
        parts.append(enc.transform(df.iloc[i:i + chunk]).astype(np.float32))
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# Base models: fit on train, export predict_proba over the 5 KABCO categories
# ---------------------------------------------------------------------------

class HistGB:
    """Gradient boosting anchor (sklearn HistGradientBoosting, CPU-friendly)."""
    name = "histgb"

    def __init__(self, max_train=1_000_000, seed=SEED):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.15, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.05, n_iter_no_change=8,
            random_state=seed, class_weight="balanced",
        )
        self.max_train = max_train
        self.seed = seed

    def fit(self, X, y):
        if len(y) > self.max_train:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(y), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self.clf.fit(X, y)
        assert list(self.clf.classes_) == [1, 2, 3, 4, 5]
        return self

    def predict_proba(self, X, chunk=500_000):
        out = [self.clf.predict_proba(X[i:i + chunk]) for i in range(0, len(X), chunk)]
        return np.concatenate(out) if len(out) > 1 else out[0]


class OrderedLogit:
    """Econometric anchor: proportional-odds ordered logit.

    Own scipy implementation with analytic gradient (no post-fit Hessian —
    statsmodels' numerical Hessian at 300k x 124 params runs for hours; we only
    need predicted probabilities, not standard errors).
    P(y <= k | x) = sigmoid(tau_k - x beta); tau ordered via cumulative exp reparam.
    """
    name = "ordered_logit"

    def __init__(self, max_train=300_000, seed=SEED, l2: float = 1e-4):
        self.max_train = max_train
        self.seed = seed
        self.l2 = l2

    @staticmethod
    def _sig(z):
        return 0.5 * (np.tanh(0.5 * z) + 1.0)

    def _taus(self, theta_t):
        t = np.empty(len(theta_t))
        t[0] = theta_t[0]
        if len(t) > 1:
            t[1:] = theta_t[0] + np.cumsum(np.exp(theta_t[1:]))
        return t

    def fit(self, X, y):
        from scipy.optimize import minimize
        if len(y) > self.max_train:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(y), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        keep = X.std(axis=0) > 1e-8
        self._keep = keep
        Xf = np.asarray(X[:, keep], dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, p = Xf.shape
        Km1 = K - 1
        # init thresholds at marginal logits, beta = 0
        freq = np.bincount(y, minlength=K + 1)[1:K + 1] / n
        cum = np.clip(np.cumsum(freq)[:Km1], 1e-4, 1 - 1e-4)
        tau0 = np.log(cum / (1 - cum))
        theta0 = np.concatenate([[tau0[0]], np.log(np.maximum(np.diff(tau0), 1e-3)), np.zeros(p)])

        yk = y  # in 1..K
        up_idx = yk - 1        # tau index for F(y), = K-1 -> +inf
        lo_idx = yk - 2        # tau index for F(y-1), = -1 -> -inf

        def nll_grad(params):
            theta_t, beta = params[:Km1], params[Km1:]
            tau = self._taus(theta_t)
            eta = Xf @ beta
            a = np.where(up_idx < Km1, np.take(tau, np.minimum(up_idx, Km1 - 1)) - eta, np.inf)
            b = np.where(lo_idx >= 0, np.take(tau, np.maximum(lo_idx, 0)) - eta, -np.inf)
            Fa, Fb = self._sig(a), self._sig(b)
            P = np.clip(Fa - Fb, 1e-12, 1.0)
            fa = np.where(np.isfinite(a), Fa * (1 - Fa), 0.0)
            fb = np.where(np.isfinite(b), Fb * (1 - Fb), 0.0)
            nll = -np.log(P).sum() / n + 0.5 * self.l2 * beta @ beta
            g_eta = (fa - fb) / P                      # dNLL/deta_i (before 1/n)
            grad_beta = Xf.T @ g_eta / n + self.l2 * beta
            # dNLL/dtau_k
            gtau = np.zeros(Km1)
            np.add.at(gtau, np.clip(up_idx, 0, Km1 - 1), np.where(up_idx < Km1, -fa / P, 0.0))
            np.add.at(gtau, np.clip(lo_idx, 0, Km1 - 1), np.where(lo_idx >= 0, fb / P, 0.0))
            gtau /= n
            gtheta = np.empty(Km1)
            gtheta[0] = gtau.sum()
            if Km1 > 1:
                suffix = np.cumsum(gtau[::-1])[::-1]
                gtheta[1:] = suffix[1:] * np.exp(theta_t[1:])
            return nll, np.concatenate([gtheta, grad_beta])

        res = minimize(nll_grad, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 300, "ftol": 1e-9})
        self._theta = res.x
        self._converged = bool(res.success)
        return self

    def predict_proba(self, X, chunk=500_000):
        Km1 = K - 1
        tau = self._taus(self._theta[:Km1])
        beta = self._theta[Km1:]
        out = []
        for i in range(0, len(X), chunk):
            eta = np.asarray(X[i:i + chunk][:, self._keep], dtype=np.float64) @ beta
            cdf = self._sig(tau[None, :] - eta[:, None])
            cdf = np.concatenate([cdf, np.ones((len(eta), 1))], axis=1)
            out.append(np.diff(np.concatenate([np.zeros((len(eta), 1)), cdf], axis=1), axis=1))
        return np.concatenate(out) if len(out) > 1 else out[0]


class MultinomialLR:
    """Cheap probabilistic baseline (cumulated multinomial logistic)."""
    name = "multinomial_lr"

    def __init__(self, max_train=500_000, seed=SEED):
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(max_iter=200, C=1.0, random_state=seed)
        self.max_train = max_train
        self.seed = seed

    def fit(self, X, y):
        if len(y) > self.max_train:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(y), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self.clf.fit(X, y)
        assert list(self.clf.classes_) == [1, 2, 3, 4, 5]
        return self

    def predict_proba(self, X, chunk=500_000):
        out = [self.clf.predict_proba(X[i:i + chunk]) for i in range(0, len(X), chunk)]
        return np.concatenate(out) if len(out) > 1 else out[0]


MODEL_REGISTRY = {m.name: m for m in (HistGB, OrderedLogit, MultinomialLR)}


# ---------------------------------------------------------------------------
# CDF cache: fit/predict once per (split, model), reuse across experiments
# ---------------------------------------------------------------------------

CACHE = ROOT / "experiments" / "cache"


def _split_fingerprint(ca, te) -> str:
    """Row-order fingerprint of a split: caches are positional arrays, so a cache is
    only valid against the exact (content AND order) of the split it was built from."""
    import hashlib
    h = hashlib.sha256()
    h.update(ca["y_kabco"].to_numpy(dtype=np.int8).tobytes())
    h.update(te["y_kabco"].to_numpy(dtype=np.int8).tobytes())
    return h.hexdigest()[:16]


def prepared_cdfs(split_name: str, model_name: str, tr, ca, te, X=None):
    """Return (cdf_ca, cdf_te, timing) with disk caching keyed by (split, model).

    X = (X_tr, X_ca, X_te) if the caller already encoded; otherwise encoded here.
    Cache stores float32 CDFs (n x 5) — small — plus fit/predict wall times and a
    row-order fingerprint; a fingerprint mismatch (e.g. cache built from a snapshot
    with different row order) raises instead of silently misaligning.
    """
    import time as _time
    CACHE.mkdir(exist_ok=True)
    fp = _split_fingerprint(ca, te)
    f = CACHE / f"{split_name}_{model_name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if "fingerprint" not in z.files:
            raise RuntimeError(
                f"{f.name}: cache predates the row-order fingerprint and cannot be "
                f"validated against the current snapshot — regenerate it (delete the "
                f"file) or restore the snapshot it was built from.")
        if str(z["fingerprint"]) != fp:
            raise RuntimeError(
                f"{f.name}: cache row-order fingerprint {z['fingerprint']} != current "
                f"split fingerprint {fp}; the snapshot row order changed — regenerate.")
        return (z["cdf_ca"], z["cdf_te"],
                {"fit_s": float(z["fit_s"]), "predict_s": float(z["predict_s"]),
                 "cached": True})
    if X is None:
        enc = make_encoder().fit(tr)
        X = (encode(enc, tr), encode(enc, ca), encode(enc, te))
    X_tr, X_ca, X_te = X
    y_tr = tr["y_kabco"].to_numpy()
    t = _time.time()
    model = MODEL_REGISTRY[model_name]().fit(X_tr, y_tr)
    fit_s = _time.time() - t
    t = _time.time()
    from choir import cdf_from_proba
    cdf_ca = cdf_from_proba(model.predict_proba(X_ca)).astype(np.float32)
    cdf_te = cdf_from_proba(model.predict_proba(X_te)).astype(np.float32)
    predict_s = _time.time() - t
    np.savez_compressed(f, cdf_ca=cdf_ca, cdf_te=cdf_te, fit_s=fit_s,
                        predict_s=predict_s, fingerprint=fp)
    return cdf_ca, cdf_te, {"fit_s": fit_s, "predict_s": predict_s, "cached": False}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def coverage_stats(y, lo, hi):
    y = np.asarray(y)
    cov = (y >= lo) & (y <= hi)
    return {
        "coverage": float(cov.mean()),
        "avg_width": float(np.mean(hi - lo + 1)),
        "contiguous_frac": 1.0,  # intervals by construction; runtime-checked in choir
        "n": int(len(y)),
        "se": float(np.sqrt(cov.mean() * (1 - cov.mean()) / len(y))),
    }
