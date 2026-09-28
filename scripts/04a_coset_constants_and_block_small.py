"""
04a -- coset dichotomy constants and block-small searches for ML-DSA-65 (q=8380417, n=256).

PART A (coset slot sets, PROVABLE): if the dependent slot set S is a coset of the subgroup of
order s (s | 256) in the group of 512-th roots, then every character t in V_S^perp satisfies the
recurrence t_{j+s} = a t_j (a = omega^s), i.e. t is a geometric progression on each residue class
mod s.  Hence B(t) = prod_j min(1, q/(2 alpha |t_j|)) <= 2^{-r M_s} for r nonzero classes, where
   M_s := -max_{x != 0} log2 prod_{m=0}^{256/s-1} min(1, q/(2 alpha |a^m x|_c)).
This script computes M_s EXACTLY (brute force over all x in Z_q^*) for every s | 256 and evaluates
   sum_{t != 0} B(t) <= (1 + (q-1) 2^{-M_s})^s - 1.

PART B (general slot sets, exhaustive search over block-small characters, phi_64; sets printed in the paper's labelling): for s < 24 a character with
more than 256 - 256/s small coordinates must be entirely small on some consecutive block J of
length s; the minor (omega_i^j)_{i in S, j in J} is a Vandermonde in omega_i times a diagonal,
hence invertible, so t is determined by t|_J.  Enumerating all t|_J in [-M, M]^s for each block
and extending by the recurrence (forward and backward) certifies max_{t != 0} B(t) over ALL
characters that could exceed the 'every block contains a large coordinate' bound.
Run:  python scripts/04a_coset_constants_and_block_small.py [A] [B3] [B4]
"""
import math, sys, time, itertools
import numpy as np
from params import q, n, alpha

zeta = 1753
def root(i): return pow(zeta, 2 * int(i) + 1, q)   # slot i root (odd power of primitive 512th root)

K_CELLS = q / (2.0 * alpha)          # = 8.0000...: phi for the cells alone (Part B, block-small runs)

def logB_vec(T, K=K_CELLS):
    """T: array (..., n) of residues; returns sum_j log2 min(1, K/|t_j|_c) along last axis."""
    tc = np.minimum(T % q, q - (T % q)); tc = np.maximum(tc, 1)
    return np.log2(np.minimum(1.0, K / tc)).sum(axis=-1)

# ----------------------------------------------------------------------------- PART A
def part_a(K=64.0):
    print(f"PART A  coset slot sets: exact M_s and the character-sum bound, phi = min(1, {K:g}/|a|_c)")
    print(f"   {'s':>4s} {'ratio order':>11s} {'M_s (bits)':>11s} {'s*M_s':>8s} {'log2 sum_t B(t) <= log2[(1+(q-1)2^-M_s)^s - 1]':>50s}")
    res = {}
    for s in (1, 2, 4, 8, 16, 32, 64, 128):
        L = n // s
        om_inv = pow(root(0), q - 2, q)
        a = pow(om_inv, s, q)
        pw = np.array([pow(a, m, q) for m in range(L)], dtype=np.int64)
        best = -1e9
        chunk = 1 << 17
        t0 = time.time()
        for start in range(1, q, chunk):
            xs = np.arange(start, min(start + chunk, q), dtype=np.int64)
            T = (xs[:, None] * pw[None, :]) % q
            best = max(best, float(logB_vec(T, K).max()))
        M = -best
        lgsum = math.log2(s * (q - 1)) - M
        res[s] = M
        print(f"   {s:4d} {512//s:11d} {M:11.1f} {s*M:8.0f} {lgsum:50.1f}   [{time.time()-t0:.0f}s]")
    print("   => for every coset S: sum over characters with r nonzero classes is C(s,r)(q-1)^r 2^{-r M_s};")
    print("      dominated by r=1 whenever M_s > log2 q = 23, i.e. for all s <= 128 above.")
    return res

# ----------------------------------------------------------------------------- PART B
STRUCT4 = [(0, 128, 64, 1), (0, 128, 1, 129), (0, 1, 2, 3), (0, 64, 128, 193)]
STRUCT3 = [(0, 128, 1), (0, 1, 2), (0, 85, 170)]

def recurrence(S):
    """monic P_S(X) = prod (X - omega_i); returns pk with t_{j+s} = sum_k pk[k] t_{j+k}."""
    s = len(S)
    oms = [pow(root(i), q - 2, q) for i in S]
    poly = [1]
    for w in oms:
        newp = [0] * (len(poly) + 1)
        for d, c in enumerate(poly):
            newp[d + 1] = (newp[d + 1] + c) % q
            newp[d] = (newp[d] - c * w) % q
        poly = newp
    return [(-poly[k]) % q for k in range(s)]

def certify(S, M):
    """max log2 B(t) over all characters supported on S that are small (|t_j|<=M) on some block."""
    s = len(S)
    pk = recurrence(S)
    pk_arr = np.array(pk, dtype=np.int64); pk_hi = pk_arr[1:]; inv0 = pow(int(pk[0]), q - 2, q)
    box = np.array(list(itertools.product(range(-M, M + 1), repeat=s)), dtype=np.int64)
    box = box[np.any(box != 0, axis=1)]
    best = -1e9
    for j0 in range(0, n - s + 1, s):
        T = np.zeros((len(box), n), dtype=np.int64)
        T[:, j0:j0 + s] = box
        st = box.copy()
        for j in range(j0 + s, n):                                   # forward: products < 2^49, exact
            nxt = (st @ pk_arr) % q
            T[:, j] = nxt
            st = np.concatenate([st[:, 1:], nxt[:, None]], axis=1)
        for j in range(j0 - 1, -1, -1):                              # backward: t_j = (t_{j+s} - sum_{k>=1} pk[k] t_{j+k}) / pk[0]
            window = T[:, j + 1:j + s]
            val = (T[:, j + s] - (window @ pk_hi) % q) % q
            T[:, j] = (val * inv0) % q
        best = max(best, float(logB_vec(T, 64.0).max()))       # phi_64, as in Section "Searches"
    return best

def part_b(s, M, n_random, seed=0):
    rng = np.random.default_rng(seed)
    print(f"PART B  general slot sets, s={s}, exhaustive over t|_J in [-{M},{M}]^s for every consecutive block J")
    sets = [tuple(int(v) for v in sorted(rng.choice(n, s, replace=False))) for _ in range(n_random)]
    sets += (STRUCT4 if s == 4 else STRUCT3 if s == 3 else [])
    worst = -1e9; worst_S = None; t0 = time.time()
    for S in sets:
        b = certify(S, M)
        tag = "STRUCT" if S in STRUCT4 + STRUCT3 else "random"
        if b > worst: worst, worst_S = b, S
        Sm = tuple(sorted(255 - i for i in S))     # this program attaches root(i)^-1 = root(255-i) to index i
        print(f"   S={str(Sm):24s} [{tag}]  (paper labelling)  max log2 B over block-small characters = {b:8.1f}")
    print(f"   worst over {len(sets)} sets: {worst:.1f} at S={tuple(sorted(255 - i for i in worst_S))} (paper labelling);   crude-bound requirement: < -(7.7*s+17) = {-(7.7*s+17):.0f};"
          f"   coset value would be about -{4390/s:.0f}   [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B3", "B4"]
    if "A" in which: part_a(64.0); part_a(K_CELLS)      # Table "coset constants" (K = 64) and the cells-only comparison
    if "B3" in which: part_b(3, M=8, n_random=30)
    if "B4" in which: part_b(4, M=4, n_random=12)
