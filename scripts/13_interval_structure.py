"""
13 -- interval structure of the output sets S_{w_j} (Lemma "Fourier factor").

For a coordinate j with shift s = (c s2)_j, |s| <= beta, and output value w_j = (v, f),
    S_{v,f} = { x in [0,q) : HighBits(x) = v, bad((x - s) mod q) = f }.
The lemma needs: for every shift, each good set S_{v,0} is ONE cyclic interval of Z_q, each bad
set S_{v,1} is a union of at most TWO cyclic intervals, and the bad sets have 6289 elements in all.
This script checks all three claims for all 393 shifts, with Decompose written from FIPS 204
(including the exceptional rule r - r0' = q - 1).
Run:  python scripts/13_interval_structure.py        (about a minute)
"""
import numpy as np
q = 8380417; gamma2 = (q - 1) // 32; alpha = 2 * gamma2; beta = 196

r = np.arange(q, dtype=np.int64)
u = r % alpha
r0 = np.where(u <= alpha // 2, u, u - alpha)              # centred representative
exc = (r - r0) == q - 1                                   # FIPS exceptional case
hb = np.where(exc, 0, (r - r0) // alpha)
lb = np.where(exc, r0 - 1, r0)
bad = np.abs(lb) >= gamma2 - beta                         # bad coordinate of the r0-test

def cyclic_runs(mask):
    """number of maximal cyclic intervals of Z_q on which mask is True."""
    if mask.all(): return 1
    if not mask.any(): return 0
    starts = np.count_nonzero(mask & ~np.roll(mask, 1))
    return int(starts)

worst_good = worst_bad = 0; sizes = set(); total_runs = set()
for s in range(-beta, beta + 1):
    b = bad[(r - s) % q]                                  # flag of x is bad((x - s) mod q)
    runs = 0
    for v in range(16):
        cell = hb == v
        g = cyclic_runs(cell & ~b); k = cyclic_runs(cell & b)
        worst_good = max(worst_good, g); worst_bad = max(worst_bad, k); runs += g + k
    sizes.add(int(np.count_nonzero(b))); total_runs.add(runs)
print(f"max cyclic intervals: good set {worst_good}, bad set {worst_bad}")
print(f"total bad elements over all shifts: {sorted(sizes)}   total intervals per shift: {sorted(total_runs)}")
assert worst_good == 1 and worst_bad <= 2 and sizes == {6289}
print("OK: phi(a) = min(1, 8/|a|_c + min(6289/q, 16/|a|_c)) is justified")
