"""Round 5, Task 7: the R-knob selection model for the channel floor (Fable Section 3b).

Replaces blanket range-widening with an explicit, anchored selection model. Let
s(y,t) = P(linked | Y=y, Ytilde=t) be the linkage probability, with bounded reported-label
dependence max_t s(y,t) / min_t s(y,t) <= R for each true y (R=1 is truth-tracking selection,
under which the forward channel is invariant). The linked-sample channel relates to the
population channel by M_link(t|y) proportional to M_pop(t|y) s(y,t), so
    M_pop(t|y) proportional to M_link(t|y) / s(y,t).
The CODES linkage measures M_link (agreement ~0.51 on the interior, disagreement concentrated on
the adjacent under-report cell, O and K reliable). Nstar needs M_pop.

The finite scenario grid uses a modal-cell concentration transform for each value of R.
This transform is a declared stress-test scenario. It is not an optimizer over every
selection function satisfying the R ratio. The script reports the minimum Nstar among the
evaluated scenarios that pass the composition-residual filter.
"""
import numpy as np
from itertools import product
from scipy.optimize import nnls

K, ALPHA, RES_TOL = 5, 0.10, 0.03
PTILDE = {"baseline": np.array([.8422, .0947, .0550, .0072, .0008]),
          "rural_highspeed": np.array([.7900, .0885, .0907, .0250, .0057]),
          "unrestrained": np.array([.3881, .1458, .2243, .1582, .0835]),
          "motorcycle": np.array([.1439, .1839, .3661, .2496, .0565])}
ACH = {"baseline": 1.4026, "rural_highspeed": 1.9524, "unrestrained": 3.5819, "motorcycle": 3.5517}


def build_link(aO, aint, aK, fd, leak):
    """Linked-sample channel: CODES-anchored. Diagonal per-category, off-diagonal tilted to the
    adjacent under-report cell by fd (the evidence: 'essentially all disagreement one category
    down'), plus a true-O/C -> reported-A leak."""
    rel = [aO, aint, aint, aint, aK]
    M = np.zeros((K, K))
    for y in range(K):
        a = rel[y]
        off = 1 - a
        dn, up = fd * off, (1 - fd) * off
        M[y, y] = a
        M[y, y - 1 if y - 1 >= 0 else y] += dn
        M[y, y + 1 if y + 1 < K else y] += up
    for y in (0, 1):
        d = leak * M[y, y]
        M[y, y] -= d
        M[y, 3] += d
    return M / M.sum(1, keepdims=True)


def to_population(M_link, R):
    """Apply the declared modal-cell concentration scenario for R."""
    M = M_link.copy()
    for y in range(K):
        w = np.ones(K)
        w[np.argmax(M[y])] = R
        M[y] = M[y] * w
    return M / M.sum(1, keepdims=True)


def invert_pA(M, pt):
    pA, r = nnls(M.T, pt)
    s = pA.sum()
    return (pA / s if s > 0 else pA), float(r)


def nstar(M, pA):
    cells = sorted(((M[y, t], y, t) for y in range(K) for t in range(K)), reverse=True)
    need, cov, sz = 1 - ALPHA, 0.0, 0.0
    for d, y, t in cells:
        ac, az = pA[y] * d, pA[y]
        if cov + ac >= need:
            return sz + ((need - cov) / ac * az if ac > 0 else 0.0)
        cov += ac
        sz += az
    return sz


# CODES-anchored linked-channel grid: interior agreement ~0.51 band, disagreement concentrated
# down (fd high, per the linkage evidence), O and K reliable.
G = dict(aO=np.linspace(.72, .92, 5), aint=np.linspace(.46, .58, 5),
         aK=np.linspace(.92, 1.0, 3), fd=np.linspace(.70, .95, 4), leak=np.linspace(0, .18, 3))
RS = [1, 2, 5]


def main():
    print(f"R-knob selection sensitivity of the channel floor, alpha={ALPHA}.")
    print("s(y,t)=P(linked|Y=y,Ytilde=t), max_t/min_t <= R per row. R=1 = truth-tracking "
          "(invariance).")
    print(f"Minimum scenario-implied Nstar per stratum over the evaluated finite grid, "
          f"using the declared modal-cell concentration transform:\n")
    hdr = f"{'stratum':16s} {'achieved':>8s} " + " ".join(f"{'R='+str(r):>12s}" for r in RS)
    print(hdr)
    combos = list(product(G['aO'], G['aint'], G['aK'], G['fd'], G['leak']))
    for s in PTILDE:
        cells = [f"{s:16s} {ACH[s]:>8.2f}"]
        for R in RS:
            ns = []
            for aO, aint, aK, fd, lk in combos:
                Mp = to_population(build_link(aO, aint, aK, fd, lk), R)
                pA, r = invert_pA(Mp, PTILDE[s])
                if r <= RES_TOL:
                    ns.append(nstar(Mp, pA))
            lo = min(ns) if ns else float("nan")
            cells.append(f"{lo:>5.2f} ({lo/ACH[s]*100:>4.1f}%)")
        print("  ".join(cells))
    print("\nRead: R=5 is heavy reported-label-dependent selection. If Nstar_lo(R=5) on the hard "
          "strata stays above the printed fraction (motorcycle >=2/5 of 3.55=1.42; unrestrained "
          ">=1/3 of 3.58=1.19), the claim survives the R-knob and is TIGHTER (higher floor) than "
          "the blanket widening, recovering floor honestly.")


if __name__ == "__main__":
    main()
