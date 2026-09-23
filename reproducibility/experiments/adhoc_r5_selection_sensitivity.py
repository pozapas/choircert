"""Round 5, Task 6: linkage-selection sensitivity of the channel floor Nstar (Fable Q4/Q3).

The channel floor uses (i) the FORWARD kernel M_A(t|y) and (ii) the POPULATION true composition
p_A. Our p_A comes from inverting the full-Texas reported composition (5.2M records), NOT the
linked sample, so it carries no linkage selection. The only selection channel is through M_A,
anchored to the Wisconsin CODES linked sample.

Direction of the risk. If linkage depends on TRUE severity alone (S _|_ Ytilde | Y), the forward
columns M_A(.|y) are selection-INVARIANT and Nstar is unbiased (Scenario A). The claim is at risk
only if linkage also depends on the REPORTED label given truth (S not _|_ Ytilde | Y): then the
linked-sample kernel is distorted, and if selection made the linked channel look NOISIER than the
population (Fable's low-end mechanism: reported-O victims are linked mainly when actually injured),
the population channel is LESS noisy, Nstar is LOWER, and the decomposition percentages fall.

The stated claims use the exact lower ends: motorcycle Nstar_lo=1.95 -> ">=55%" of 3.55,
unrestrained 1.51 -> ">=42%" of 3.58. These sit at the boundary, so any downward movement breaks
the exact percentages. This script recomputes Nstar_lo under a WIDENED kernel set that brackets the
selection uncertainty in the dangerous (population-less-noisy) direction, and reports the
selection-robust percentage that survives.
"""
import numpy as np
from itertools import product

PTILDE = {
    "baseline":        np.array([0.8422, 0.0947, 0.0550, 0.0072, 0.0008]),
    "rural_highspeed": np.array([0.7900, 0.0885, 0.0907, 0.0250, 0.0057]),
    "unrestrained":    np.array([0.3881, 0.1458, 0.2243, 0.1582, 0.0835]),
    "motorcycle":      np.array([0.1439, 0.1839, 0.3661, 0.2496, 0.0565]),
}
ACHIEVED = {"baseline": 1.4026, "rural_highspeed": 1.9524,
            "unrestrained": 3.5819, "motorcycle": 3.5517}
ALPHA, K, RES_TOL = 0.10, 5, 0.03


def build_M(a_O, a_int, a_K, fd, leak):
    rel = [a_O, a_int, a_int, a_int, a_K]
    M = np.zeros((K, K))
    for y in range(K):
        a = rel[y]
        off = 1 - a
        dn, up = fd * off, (1 - fd) * off
        M[y, y] = a
        if y - 1 >= 0:
            M[y, y - 1] += dn
        else:
            M[y, y] += dn
        if y + 1 < K:
            M[y, y + 1] += up
        else:
            M[y, y] += up
    for y in (0, 1):
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
            return size + ((need - cov) / ac * asz if ac > 0 else 0.0)
        cov += ac
        size += asz
    return size


def nstar_lohi(gridspec, stratum):
    combos = product(*[gridspec[k] for k in
                       ("a_O", "a_int", "a_K", "fd", "leak")])
    ns = []
    for a_O, a_int, a_K, fd, leak in combos:
        M = build_M(a_O, a_int, a_K, fd, leak)
        pA, res = invert_pA(M, PTILDE[stratum])
        if res <= RES_TOL:
            ns.append(nstar(M, pA))
    return (min(ns), max(ns), len(ns)) if ns else (None, None, 0)


# Scenario A: the declared identified set (linkage-invariant forward kernel; S _|_ Ytilde | Y).
BASE = dict(a_O=np.linspace(0.70, 0.95, 6), a_int=np.linspace(0.44, 0.58, 6),
            a_K=np.linspace(0.90, 1.00, 3), fd=np.linspace(0.50, 0.70, 3),
            leak=np.linspace(0.00, 0.15, 3))
# Scenario B: widen in the DANGEROUS direction -- population channel possibly LESS noisy than the
# linked sample (higher agreement), which lowers Nstar. Push interior and a_O agreement UP, and
# also allow the low end to stay noisy (Fable's inflation) by keeping the floor of the ranges.
WIDE = dict(a_O=np.linspace(0.70, 0.97, 7), a_int=np.linspace(0.44, 0.66, 8),
            a_K=np.linspace(0.90, 1.00, 3), fd=np.linspace(0.45, 0.72, 4),
            leak=np.linspace(0.00, 0.20, 4))


def main():
    print(f"Linkage-selection sensitivity of Nstar, alpha={ALPHA}.")
    print("Scenario A = declared identified set (forward kernel selection-invariant).")
    print("Scenario B = widened for reported-label-dependent selection (population less noisy).\n")
    print(f"{'stratum':16s} {'achieved':>8s} | {'A: Nstar_lo':>11s} {'A share':>8s} | "
          f"{'B: Nstar_lo':>11s} {'B share':>8s}  {'claim survives':>14s}")
    for s in ["baseline", "rural_highspeed", "unrestrained", "motorcycle"]:
        loA, hiA, nA = nstar_lohi(BASE, s)
        loB, hiB, nB = nstar_lohi(WIDE, s)
        ach = ACHIEVED[s]
        shA = loA / ach
        shB = loB / ach
        # claim survives if B's lower end still gives a meaningful floor (>1) and a share we can state
        surv = ("n/a (informative)" if loB <= 1.05
                else f">= {np.floor(shB*20)/20*100:.0f}%")
        print(f"{s:16s} {ach:>8.3f} | {loA:>11.3f} {shA*100:>7.1f}% | "
              f"{loB:>11.3f} {shB*100:>7.1f}%  {surv:>14s}")
    print("\nRead: 'B: Nstar_lo' is the selection-robust floor. If it stays well above 1 on the "
          "hard strata, the decomposition survives selection; the stated percentage should be the "
          "conservative 'B share' rounded DOWN, not the boundary Scenario-A value.")


if __name__ == "__main__":
    main()
