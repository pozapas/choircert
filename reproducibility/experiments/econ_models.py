"""Econometric-canon base models (CHOIR_framework.md 3.3 items 2-3).

LCOrderedLogit: latent-class ordered logit with covariate gating (concomitant
variables), fit by EM on a training subsample. Exports predicted CDFs and the MAP
gate class — the 'classical partition' variant for Theorem 2 / E3.

RPOrderedLogit: random-parameters ordered logit (random intercept + one random
slope, normal heterogeneity), simulated MLE with Halton draws. We certify the
canon, we do not compete with it: both models are wrapped by the identical
certification layer; their probabilities are cumulated and scored like any other.

Descriptive use only — coefficients are never interpreted causally anywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri

K = 5
SEED = 20260704


def _sig(z):
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


def _taus(theta_t):
    t = np.empty(len(theta_t))
    t[0] = theta_t[0]
    if len(t) > 1:
        t[1:] = theta_t[0] + np.cumsum(np.exp(theta_t[1:]))
    return t


def _ol_nll_grad(params, Xf, y, w, l2):
    """Weighted proportional-odds NLL and analytic gradient (see common.OrderedLogit)."""
    n, p = Xf.shape
    Km1 = K - 1
    theta_t, beta = params[:Km1], params[Km1:]
    tau = _taus(theta_t)
    eta = Xf @ beta
    up_idx = y - 1
    lo_idx = y - 2
    a = np.where(up_idx < Km1, np.take(tau, np.minimum(up_idx, Km1 - 1)) - eta, np.inf)
    b = np.where(lo_idx >= 0, np.take(tau, np.maximum(lo_idx, 0)) - eta, -np.inf)
    Fa, Fb = _sig(a), _sig(b)
    P = np.clip(Fa - Fb, 1e-12, 1.0)
    fa = np.where(np.isfinite(a), Fa * (1 - Fa), 0.0)
    fb = np.where(np.isfinite(b), Fb * (1 - Fb), 0.0)
    W = w.sum()
    nll = -(w * np.log(P)).sum() / W + 0.5 * l2 * beta @ beta
    g_eta = w * (fa - fb) / P
    grad_beta = Xf.T @ g_eta / W + l2 * beta
    gtau = np.zeros(Km1)
    np.add.at(gtau, np.clip(up_idx, 0, Km1 - 1), np.where(up_idx < Km1, -w * fa / P, 0.0))
    np.add.at(gtau, np.clip(lo_idx, 0, Km1 - 1), np.where(lo_idx >= 0, w * fb / P, 0.0))
    gtau /= W
    gtheta = np.empty(Km1)
    gtheta[0] = gtau.sum()
    if Km1 > 1:
        suffix = np.cumsum(gtau[::-1])[::-1]
        gtheta[1:] = suffix[1:] * np.exp(theta_t[1:])
    return nll, np.concatenate([gtheta, grad_beta])


def fit_ol_weighted(Xf, y, w, l2=1e-4, x0=None, maxiter=200):
    n = len(y)
    Km1 = K - 1
    if x0 is None:
        freq = np.bincount(y, weights=w, minlength=K + 1)[1:K + 1] / w.sum()
        cum = np.clip(np.cumsum(freq)[:Km1], 1e-4, 1 - 1e-4)
        tau0 = np.log(cum / (1 - cum))
        x0 = np.concatenate([[tau0[0]], np.log(np.maximum(np.diff(tau0), 1e-3)),
                             np.zeros(Xf.shape[1])])
    res = minimize(_ol_nll_grad, x0, args=(Xf, y, w, l2), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-9})
    return res.x


def _ol_proba(params, Xf):
    Km1 = K - 1
    tau = _taus(params[:Km1])
    eta = Xf @ params[Km1:]
    cdf = _sig(tau[None, :] - eta[:, None])
    cdf = np.concatenate([cdf, np.ones((len(eta), 1))], axis=1)
    return np.diff(np.concatenate([np.zeros((len(eta), 1)), cdf], axis=1), axis=1)


class LCOrderedLogit:
    """EM latent-class ordered logit, gate = multinomial logistic on covariates."""
    name = "lc_logit"

    def __init__(self, n_classes=3, max_train=300_000, em_iters=5, seed=SEED):
        self.C = n_classes
        self.max_train = max_train
        self.em_iters = em_iters
        self.seed = seed

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(self.seed)
        if len(y) > self.max_train:
            idx = rng.choice(len(y), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        keep = X.std(axis=0) > 1e-8
        self._keep = keep
        Xf = np.asarray(X[:, keep], dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n = len(y)

        # init: single OL, split by linear-index terciles, softened
        base = fit_ol_weighted(Xf, y, np.ones(n))
        eta = Xf @ base[K - 1:]
        qs = np.quantile(eta, np.linspace(0, 1, self.C + 1)[1:-1])
        hard = np.digitize(eta, qs)
        R = np.full((n, self.C), 0.1 / (self.C - 1))
        R[np.arange(n), hard] = 0.9

        self.experts_ = [None] * self.C
        # max_iter raised 200 -> 2000 (2026-07-16). At 200 this gate raised
        # ConvergenceWarning on every EM iteration, so its coefficients sat wherever lbfgs
        # happened to stop and were BLAS/thread-sensitive rather than determined by the
        # data. That is the same defect fixed the same day at the four density-ratio sites
        # in e5_temporal.py / e6_spatial.py, and it matters more here: this gate defines
        # the latent-class partition that the paper reports as its most EFFICIENT
        # partition (mean width 3.236, the narrowest of the four tested). An efficiency
        # claim resting on an unconverged optimizer is not reproducible across machines.
        # If it still warns at 2000, raise it further rather than silencing the warning.
        gate = LogisticRegression(max_iter=2000, C=1.0, random_state=self.seed)
        prev_ll = -np.inf
        for it in range(self.em_iters):
            # M-step: weighted OL per class (warm start), gate on responsibilities
            for c in range(self.C):
                self.experts_[c] = fit_ol_weighted(
                    Xf, y, R[:, c] + 1e-6, x0=self.experts_[c], maxiter=100)
            rep_idx = rng.choice(n, min(n, 150_000), replace=False)
            Xg = np.tile(Xf[rep_idx], (self.C, 1))
            yg = np.repeat(np.arange(self.C), len(rep_idx))
            wg = np.concatenate([R[rep_idx, c] for c in range(self.C)])
            gate.fit(Xg, yg, sample_weight=wg + 1e-8)
            # E-step
            pi = gate.predict_proba(Xf)
            like = np.stack([_ol_proba(self.experts_[c], Xf)[np.arange(n), y - 1]
                             for c in range(self.C)], axis=1)
            joint = np.clip(pi * like, 1e-300, None)
            ll = np.log(joint.sum(1)).mean()
            R = joint / joint.sum(1, keepdims=True)
            print(f"  LC EM iter {it}: loglik/n={ll:.5f} "
                  f"gate share={pi.mean(0).round(3)}", flush=True)
            if ll - prev_ll < 1e-5 and it >= 2:
                break
            prev_ll = ll
        self.gate_ = gate
        return self

    def _pi(self, X):
        return self.gate_.predict_proba(np.asarray(X[:, self._keep], dtype=np.float64))

    def predict_proba(self, X, chunk=500_000):
        out = []
        for i in range(0, len(X), chunk):
            Xf = np.asarray(X[i:i + chunk][:, self._keep], dtype=np.float64)
            pi = self.gate_.predict_proba(Xf)
            p = sum(pi[:, c:c + 1] * _ol_proba(self.experts_[c], Xf)
                    for c in range(self.C))
            out.append(p)
        return np.concatenate(out) if len(out) > 1 else out[0]

    def predict_classes(self, X, chunk=500_000):
        out = [self._pi(X[i:i + chunk]).argmax(1).astype(np.int16)
               for i in range(0, len(X), chunk)]
        return np.concatenate(out) if len(out) > 1 else out[0]


class RPOrderedLogit:
    """Random-parameters ordered logit: eta = x'beta + s0*z0 + s1*z1*x_rp,
    (z0, z1) ~ N(0, I) via Halton draws; simulated MLE with analytic gradient."""
    name = "rp_logit"

    def __init__(self, rp_col=0, n_draws=32, max_train=100_000, seed=SEED, l2=1e-4):
        self.rp_col = rp_col          # column index (in kept features) with random slope
        self.R = n_draws
        self.max_train = max_train
        self.seed = seed
        self.l2 = l2

    def _draws(self):
        from scipy.stats import qmc
        h = qmc.Halton(d=2, scramble=True, seed=self.seed).random(self.R)
        return ndtri(np.clip(h, 1e-6, 1 - 1e-6))  # (R, 2) standard normal

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        if len(y) > self.max_train:
            idx = rng.choice(len(y), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        keep = X.std(axis=0) > 1e-8
        self._keep = keep
        Xf = np.asarray(X[:, keep], dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, p = Xf.shape
        Km1 = K - 1
        Z = self._draws()                      # (R, 2)
        xs = Xf[:, self.rp_col]

        up_idx, lo_idx = y - 1, y - 2

        def nll_grad(params):
            theta_t = params[:Km1]
            beta = params[Km1:Km1 + p]
            ls0, ls1 = params[-2], params[-1]
            s0, s1 = np.exp(ls0), np.exp(ls1)
            tau = _taus(theta_t)
            eta0 = Xf @ beta
            L = np.zeros(n)
            acc = {"eta": np.zeros(n), "tau": np.zeros((n, Km1)),
                   "s0": np.zeros(n), "s1": np.zeros(n)}
            Pr_all = []
            for r in range(self.R):
                off = s0 * Z[r, 0] + s1 * Z[r, 1] * xs
                eta = eta0 + off
                a = np.where(up_idx < Km1,
                             np.take(tau, np.minimum(up_idx, Km1 - 1)) - eta, np.inf)
                b = np.where(lo_idx >= 0,
                             np.take(tau, np.maximum(lo_idx, 0)) - eta, -np.inf)
                Fa, Fb = _sig(a), _sig(b)
                P = np.clip(Fa - Fb, 1e-12, 1.0)
                fa = np.where(np.isfinite(a), Fa * (1 - Fa), 0.0)
                fb = np.where(np.isfinite(b), Fb * (1 - Fb), 0.0)
                L += P
                dP_deta = -(fa - fb)
                acc["eta"] += dP_deta
                acc["s0"] += dP_deta * Z[r, 0] * s0
                acc["s1"] += dP_deta * Z[r, 1] * xs * s1
                # dP/dtau_k: +fa at up_idx, -fb at lo_idx
                Pr_all.append((fa, fb))
            L /= self.R
            nll = -np.log(L).mean() + 0.5 * self.l2 * beta @ beta
            denom = n * self.R * L
            grad_beta = -(Xf.T @ (acc["eta"] / denom)) + self.l2 * beta
            g_s0 = -(acc["s0"] / denom).sum()
            g_s1 = -(acc["s1"] / denom).sum()
            gtau = np.zeros(Km1)
            for fa, fb in Pr_all:
                np.add.at(gtau, np.clip(up_idx, 0, Km1 - 1),
                          np.where(up_idx < Km1, -fa / denom, 0.0))
                np.add.at(gtau, np.clip(lo_idx, 0, Km1 - 1),
                          np.where(lo_idx >= 0, fb / denom, 0.0))
            gtheta = np.empty(Km1)
            gtheta[0] = gtau.sum()
            if Km1 > 1:
                suffix = np.cumsum(gtau[::-1])[::-1]
                gtheta[1:] = suffix[1:] * np.exp(theta_t[1:])
            return nll, np.concatenate([gtheta, grad_beta, [g_s0, g_s1]])

        x0 = np.concatenate([fit_ol_weighted(Xf, y, np.ones(n)),
                             [np.log(0.5), np.log(0.1)]])
        res = minimize(nll_grad, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 250, "ftol": 1e-9})
        self._theta = res.x
        self._converged = bool(res.success)
        self._p = p
        print(f"  RP-OL converged={res.success} "
              f"sigma0={np.exp(res.x[-2]):.3f} sigma1={np.exp(res.x[-1]):.3f}", flush=True)
        return self

    def predict_proba(self, X, chunk=200_000):
        Km1 = K - 1
        theta_t = self._theta[:Km1]
        beta = self._theta[Km1:Km1 + self._p]
        s0, s1 = np.exp(self._theta[-2]), np.exp(self._theta[-1])
        tau = _taus(theta_t)
        Z = self._draws()
        out = []
        for i in range(0, len(X), chunk):
            Xf = np.asarray(X[i:i + chunk][:, self._keep], dtype=np.float64)
            eta0 = Xf @ beta
            xs = Xf[:, self.rp_col]
            p_acc = np.zeros((len(Xf), K))
            for r in range(self.R):
                eta = eta0 + s0 * Z[r, 0] + s1 * Z[r, 1] * xs
                cdf = _sig(tau[None, :] - eta[:, None])
                cdf = np.concatenate([cdf, np.ones((len(eta), 1))], axis=1)
                p_acc += np.diff(np.concatenate([np.zeros((len(eta), 1)), cdf], axis=1),
                                 axis=1)
            out.append(p_acc / self.R)
        return np.concatenate(out) if len(out) > 1 else out[0]
