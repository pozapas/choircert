"""Round 5, Task 5: the Nstar identified set (Molinari 2008 direct-misclassification approach).

Evaluates a finite grid of declared reporting-channel scenarios. The published CODES
numbers are marginal rather than a full 5x5 joint, so the forward kernel M_A is not
identified. The reported minima and maxima summarize the evaluated grid. They are not
endpoints of a continuously optimized identified set.

Parameterization of the within-stratum kernel M_A (Fable: kernel is stratum-specific), by
interpretable knobs with ranges anchored to the published evidence:
  a_O   in [0.70, 0.95]  reliability of reported O (minor crashes; boundary, cannot under-report)
  a_int in [0.44, 0.58]  interior C/B/A agreement, ~0.51 population disagreement +/- a band
  a_K   in [0.90, 1.00]  fatal near-exact
  fd    in [0.50, 0.70]  fraction of off-diagonal mass that is UNDER-reporting (net tilt down)
  leak  in [0.00, 0.15]  two-up over-report of true O/C to reported A (the 38.2% direction)
Consistency filter: a kernel is admitted for a stratum only if ptildeA = M^T pA has a simplex
solution pA with inversion residual below RES_TOL (the observed reported composition must be
reachable through M). Nstar over the ADMITTED kernels gives the identified interval.

Reports, per stratum: the Nstar interval, whether min Nstar > 1 (Corollary A1 needs this on the
hard strata), and the specification-test margin against the min achieved Mondrian base width.
"""
import numpy as np
from itertools import product

PTILDE = {
    "baseline":        np.array([0.8422, 0.0947, 0.0550, 0.0072, 0.0008]),
    "rural_highspeed": np.array([0.7900, 0.0885, 0.0907, 0.0250, 0.0057]),
    "unrestrained":    np.array([0.3881, 0.1458, 0.2243, 0.1582, 0.0835]),
    "motorcycle":      np.array([0.1439, 0.1839, 0.3661, 0.2496, 0.0565]),
}
ACHIEVED_MIN = {"baseline": 1.4026, "rural_highspeed": 1.9524,
                "unrestrained": 3.5819, "motorcycle": 3.5517}
ALPHA, K, RES_TOL = 0.10, 5, 0.03


def build_M(a_O, a_int, a_K, fd, leak):
    rel = [a_O, a_int, a_int, a_int, a_K]      # 0=O,1=C,2=B,3=A,4=K
    M = np.zeros((K, K))
    for y in range(K):
        a = rel[y]
        off = 1 - a
        dn, up = fd * off, (1 - fd) * off
        M[y, y] = a
        (M.__setitem__((y, y - 1), M[y, y - 1] + dn) if y - 1 >= 0
         else M.__setitem__((y, y), M[y, y] + dn))
        (M.__setitem__((y, y + 1), M[y, y + 1] + up) if y + 1 < K
         else M.__setitem__((y, y), M[y, y] + up))
    for y in (0, 1):                            # true O/C leak up to reported A
        d = leak * M[y, y]
        M[y, y] -= d
        M[y, 3] += d
    return M / M.sum(axis=1, keepdims=True)


def invert_pA(M, ptildeA):
    from scipy.optimize import nnls
    pA, res = nnls(M.T, ptildeA)
    s = pA.sum()
    return (pA / s if s > 0 else pA), float(res)


def nstar(M, pA, alpha=ALPHA):
    cells = sorted(((M[y, t], y, t) for y in range(K) for t in range(K)), reverse=True)
    need, cov, size = 1 - alpha, 0.0, 0.0
    for dens, y, t in cells:
        ac, asz = pA[y] * dens, pA[y]
        if cov + ac >= need:
            size += (need - cov) / ac * asz if ac > 0 else 0.0
            return size
        cov += ac
        size += asz
    return size


def main():
    grid = dict(
        a_O=np.linspace(0.70, 0.95, 6), a_int=np.linspace(0.44, 0.58, 6),
        a_K=np.linspace(0.90, 1.00, 3), fd=np.linspace(0.50, 0.70, 3),
        leak=np.linspace(0.00, 0.15, 3))
    combos = list(product(grid["a_O"], grid["a_int"], grid["a_K"], grid["fd"], grid["leak"]))
    print(f"Nstar scenario range over {len(combos)} evaluated channel scenarios "
          f"(Molinari-style), alpha={ALPHA}, residual tol={RES_TOL}.")
    print(f"{'stratum':16s} {'Nstar_lo':>9s} {'Nstar_hi':>9s} {'#admit':>7s} "
          f"{'achieved':>9s} {'min>1?':>7s} {'spec-test':>18s}")
    for s in ["baseline", "rural_highspeed", "unrestrained", "motorcycle"]:
        ns, admitted = [], 0
        for a_O, a_int, a_K, fd, leak in combos:
            M = build_M(a_O, a_int, a_K, fd, leak)
            pA, res = invert_pA(M, PTILDE[s])
            if res <= RES_TOL:
                admitted += 1
                ns.append(nstar(M, pA))
        if not ns:
            print(f"{s:16s}  NO admitted kernels (all residuals > {RES_TOL}) -- observed "
                  f"reported composition unreachable through this family; kernel misspecified.")
            continue
        lo, hi = min(ns), max(ns)
        ach = ACHIEVED_MIN[s]
        # spec test: is the achieved width below the WHOLE identified interval's lower end?
        verdict = ("REFUTED (achieved < Nstar_lo)" if ach < lo - 1e-9
                   else "consistent" if ach >= lo else "boundary")
        print(f"{s:16s} {lo:>9.3f} {hi:>9.3f} {admitted:>7d} {ach:>9.3f} "
              f"{'yes' if lo > 1 else 'NO':>7s} {verdict:>18s}")
    print("\nRead: on the HARD strata, Nstar_lo > 1 supports Corollary A1 (irreducible noise "
          "floor), and Nstar_lo < achieved means the theorem is consistent (not pre-refuted). "
          "Baseline REFUTED under the low-agreement part of the set is the expected signal that "
          "baseline reporting is more reliable than the interior -- report the admitted set.")


if __name__ == "__main__":
    main()
