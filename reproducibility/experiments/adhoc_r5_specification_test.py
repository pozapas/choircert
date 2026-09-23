"""Round 5, Task 4: build the misreporting kernel and run Theorem A's specification test.

Fable's mandate (FABLE_RESPONSE_thmA.md Q7): Theorem A says no valid predictor of the REPORTED
label on stratum A has mean width below Nstar(alpha, M, pA). We PUBLISH achieved widths. If any
achieved, validly-covering width is below the computed Nstar, the declared kernel is REFUTED.
Run it before writing the theorem. Baseline first (Fable predicts it binds).

Nstar = fractional knapsack LP (the CORRECTED floor): cells (y,t) sorted by M(t|y), greedy-include
to coverage 1-alpha, one fractional boundary cell. Nstar = sum of pA(y) over included cells.

pA (TRUE composition) is recovered from the observed REPORTED composition ptildeA via
ptildeA = M pA, solved on the simplex (nnls + renormalize); a large residual or a boundary
solution is itself evidence the kernel is misspecified.

We compute Nstar over a FAMILY of declared kernels (the published CODES numbers are marginal, not
a full 5x5 joint, so M is only partially identified): a uniform-agreement family (agreement a
swept) and a category-dependent family (O, K reliable; C, B, A noisy), the latter being the
kernel the paper's Remark 'category-dependent bands' already argues for. Report Nstar per stratum
for each, and the specification-test verdict against the MINIMUM achieved Mondrian base width.
"""
import numpy as np

# ---- observed REPORTED composition per stratum (from the snapshot; ptildeA) ----
PTILDE = {
    "baseline":        np.array([0.8422, 0.0947, 0.0550, 0.0072, 0.0008]),
    "rural_highspeed": np.array([0.7900, 0.0885, 0.0907, 0.0250, 0.0057]),
    "unrestrained":    np.array([0.3881, 0.1458, 0.2243, 0.1582, 0.0835]),
    "motorcycle":      np.array([0.1439, 0.1839, 0.3661, 0.2496, 0.0565]),
}
# ---- MINIMUM achieved Mondrian base (reported-label) width per stratum, across the six
#      non-histgb models (PC_RESULTS_4 E3 table). histgb excluded as the known outlier. ----
ACHIEVED_MIN = {   # (model attaining it)
    "baseline":        (1.4026, "dlcon"),
    "rural_highspeed": (1.9524, "dlcon"),
    "unrestrained":    (3.5819, "dlcon"),
    "motorcycle":      (3.5517, "dlcon"),
}
ALPHA = 0.10
K = 5


def kernel_uniform(a):
    """Forward M(t|y)=P(report t|true y): diagonal agreement a, remainder to adjacent +/-1
    with a net under-reporting tilt (down gets 0.6 of off-diag), boundaries absorb."""
    M = np.zeros((K, K))
    for y in range(K):           # y is true (0-indexed)
        off = 1 - a
        dn, up = 0.60 * off, 0.40 * off
        M[y, y] = a
        if y - 1 >= 0:
            M[y, y - 1] += dn
        else:
            M[y, y] += dn        # O cannot be under-reported below itself
        if y + 1 < K:
            M[y, y + 1] += up
        else:
            M[y, y] += up
    return M  # rows = true y, cols = reported t


def kernel_catdep(a_interior=0.51, a_O=0.90, a_K=0.98, a_over=0.10):
    """Category-dependent forward kernel (the paper's rem:catdep structure):
    O and K reliable, interior C/B/A carry the 0.49-disagreement, plus a two-down OVER-report
    channel calibrated to the CODES '38.2% of reported-A are true-O' fact (here as forward mass
    from true O/C up to reported A). a_over tunes that far-over-report leakage."""
    M = np.zeros((K, K))
    rel = {0: a_O, 1: a_interior, 2: a_interior, 3: a_interior, 4: a_K}  # 4==A? no: index 4 is K
    # indices: 0=O,1=C,2=B,3=A,4=K
    rel = {0: a_O, 1: a_interior, 2: a_interior, 3: a_interior, 4: a_K}
    for y in range(K):
        a = rel[y]
        off = 1 - a
        dn, up = 0.60 * off, 0.40 * off
        M[y, y] = a
        if y - 1 >= 0:
            M[y, y - 1] += dn
        else:
            M[y, y] += dn
        if y + 1 < K:
            M[y, y + 1] += up
        else:
            M[y, y] += up
    # far over-report: some true O(0) and C(1) reported as A(3) (the 38.2% direction)
    for y in (0, 1):
        leak = a_over * M[y, y]
        M[y, y] -= leak
        M[y, 3] += leak
    return M / M.sum(axis=1, keepdims=True)


def invert_pA(M, ptildeA):
    """Solve ptildeA = M^T pA for pA on the simplex (M rows=true, so ptilde = M^T pA).
    nnls then renormalize; report residual and whether it left the simplex materially."""
    from scipy.optimize import nnls
    A = M.T                      # columns of M^T are the reported dists per true y
    pA, res = nnls(A, ptildeA)
    s = pA.sum()
    pA_n = pA / s if s > 0 else pA
    return pA_n, float(res), float(s)


def nstar(M, pA, alpha):
    """Fractional-knapsack floor: cells (y,t) by density M[y,t] desc, cost pA[y], accumulate
    coverage sum pA[y]M[y,t] to 1-alpha; Nstar = accumulated cost (fractional last cell)."""
    cells = [(M[y, t], y, t) for y in range(K) for t in range(K)]
    cells.sort(reverse=True)     # highest coverage-density first
    need = 1 - alpha
    cov = 0.0
    size = 0.0
    for dens, y, t in cells:
        add_cov = pA[y] * dens
        add_size = pA[y]
        if cov + add_cov >= need:
            frac = (need - cov) / add_cov if add_cov > 0 else 0.0
            size += frac * add_size
            cov = need
            break
        cov += add_cov
        size += add_size
    return size


def run(label, kernel_fn, **kw):
    M = kernel_fn(**kw)
    print(f"\n=== {label} ===")
    print(f"{'stratum':16s} {'pA_resid':>9s} {'Nstar':>7s} {'achieved_min':>13s} "
          f"{'verdict':>10s}")
    for s in ["baseline", "rural_highspeed", "unrestrained", "motorcycle"]:
        pA, res, mass = invert_pA(M, PTILDE[s])
        N = nstar(M, pA, ALPHA)
        ach, model = ACHIEVED_MIN[s]
        verdict = "REFUTED" if N > ach + 1e-9 else "ok"
        print(f"{s:16s} {res:>9.4f} {N:>7.3f} {ach:>10.3f}({model[:3]}) {verdict:>10s}"
              + (f"   Nstar {N:.3f} > achieved {ach:.3f}" if verdict == "REFUTED" else ""))


if __name__ == "__main__":
    print("Theorem A specification test. Nstar > achieved width  =>  declared kernel REFUTED.")
    print(f"alpha={ALPHA}; achieved = min Mondrian base width over 6 non-histgb models "
          f"(PC_RESULTS_4).")
    # 1) uniform-agreement family, swept
    for a in (0.51, 0.60, 0.70, 0.80):
        run(f"UNIFORM kernel, agreement a={a}", kernel_uniform, a=a)
    # 2) category-dependent family (O/K reliable), a few settings
    run("CATEGORY-DEPENDENT kernel (O=0.90,K=0.98,interior=0.51,over=0.10)", kernel_catdep)
    run("CATEGORY-DEPENDENT kernel (O=0.85,K=0.98,interior=0.51,over=0.15)",
        kernel_catdep, a_O=0.85, a_over=0.15)
    run("CATEGORY-DEPENDENT kernel (O=0.90,K=0.98,interior=0.45,over=0.10)",
        kernel_catdep, a_interior=0.45)
