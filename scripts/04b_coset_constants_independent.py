"""
04b -- independent re-derivation of the coset constants (written from
scratch; shares no code with 04a).

Conventions here (deliberately different from 04a):
  * slot i has root r_i = zeta^(2i+1) (no inverses anywhere);
  * a character supported on slot set S is  t_j = sum_{i in S} c_i * r_i^j  (positive powers);
  * B(t) = prod_j min(1, R/|t_j|_c) with R = q/(2 alpha) = 8 + 1/alpha... computed as a float.

(a) Coset constants.  For s | 256 the coset {i0 + m*(256/s)} has roots r_i = r_{i0} * g^m with
    g = zeta^(512/s) an s-th root of unity.  Claim: characters on this coset satisfy
    t_{j+s} = a * t_j with a = r_{i0}^s.  We (1) CHECK this numerically for random c (method 1,
    direct sums), (2) compute M_s = -max_x log2 prod_{m<256/s} min(1, R/|a^m x|) by brute force
    over ALL x (method 2), and (3) CROSS-CHECK method 2 against direct full-vector evaluation of
    the sparse DFT-delta characters for 3000 random x (method 1).
(c) One general slot set at s=3, S=(0,128,1): enumerate all t|_J in [-8,8]^3 on every consecutive
    block J, recover c by solving the 3x3 Vandermonde system, evaluate the full character by
    direct sums, and report the maximum of log2 B.  Also report the same for S=(4,10,19).
"""
import itertools, math, sys, time
import numpy as np

q, n, alpha = 8380417, 256, 523776
ZETA = 1753
import os
R = float(os.environ.get("PHI_K", "64"))   # 64: Table "coset constants"; PHI_K=8.0000019 reproduces the cells-only constants

def r_of(i): return pow(ZETA, 2 * int(i) + 1, q)

def logB(T):
    Tm = T % q
    tc = np.minimum(Tm, q - Tm).astype(np.float64)
    tc = np.maximum(tc, 1.0)
    return np.log2(np.minimum(1.0, R / tc)).sum(axis=-1)

def char_direct(S, c):
    """t_j = sum_i c_i r_i^j for j in 0..n-1, exact mod q (python ints)."""
    t = np.zeros(n, dtype=np.int64)
    for ci, i in zip(c, S):
        r = r_of(i); pw = 1; col = np.empty(n, dtype=np.int64)
        for j in range(n):
            col[j] = pw; pw = pw * r % q
        t = (t + (int(ci) % q) * col) % q
    return t

def part_a():
    print("(a) coset constants, fresh derivation")
    rng = np.random.default_rng(12345)
    out = {}
    for s in (1, 2, 4, 8, 16, 32, 64, 128):
        step = n // s; i0 = 3
        S = [(i0 + m * step) % n for m in range(s)]
        a = pow(r_of(i0), s, q)
        # (1) recurrence check on 5 random characters (direct sums)
        ok = True
        for _ in range(5):
            c = rng.integers(1, q, size=s)
            t = char_direct(S, c)
            for j in range(0, n - s):
                if (t[j + s] - a * int(t[j])) % q != 0: ok = False; break
            if not ok: break
        # (2) brute-force M_s over all x via the progression x, a x, a^2 x, ...
        L = n // s
        pw = np.array([pow(a, m, q) for m in range(L)], dtype=np.int64)
        best = -1e9; chunk = 1 << 17; t0 = time.time()
        for start in range(1, q, chunk):
            xs = np.arange(start, min(start + chunk, q), dtype=np.int64)
            best = max(best, float(logB((xs[:, None] * pw[None, :]) % q).max()))
        M = -best
        # (3) cross-check: sparse DFT-delta character c_i = x * r_i^{-j0} makes t supported on class j0 mod s
        maxdiff = 0.0; support_ok = True
        for _ in range(3000 if s <= 16 else 300):
            x = int(rng.integers(1, q)); j0 = int(rng.integers(0, s))
            c = [x * pow(r_of(i), (q - 1 - j0) % (q - 1), q) % q for i in S]   # r_i^{-j0} = r_i^{q-1-j0}
            t = char_direct(S, c)
            nz = np.flatnonzero(t)
            if not np.all(nz % s == j0): support_ok = False
            direct = float(logB(t))
            prog = float(logB((x * pw * s) % q))            # class values: sum over coset of x r_i^{j-j0} = s * x * a^m
            maxdiff = max(maxdiff, abs(direct - prog))
        out[s] = M
        print(f"    s={s:3d}: recurrence t_(j+s)=a t_j holds: {ok};  M_s = {M:8.1f};  s*M_s = {s*M:6.0f};  "
              f"sparse-character support on one class: {support_ok};  |direct - progression| max = {maxdiff:.2e}   [{time.time()-t0:.0f}s]")
    return out

def part_c(S, M=8):
    print(f"(c) general slot set S={S}, s={len(S)}: exhaustive over t|_J in [-{M},{M}]^s per consecutive block, Vandermonde solve + direct evaluation")
    s = len(S); roots = [r_of(i) for i in S]
    powers = np.array([[pow(r, j, q) for j in range(n)] for r in roots], dtype=np.int64)   # s x n
    box = np.array(list(itertools.product(range(-M, M + 1), repeat=s)), dtype=np.int64)
    box = box[np.any(box != 0, axis=1)]
    def inv_mat(V):
        m = len(V); A = [[int(v) % q for v in row] + [1 if r == cc else 0 for cc in range(m)] for r, row in enumerate(V)]
        for col in range(m):
            piv = next(r for r in range(col, m) if A[r][col]); A[col], A[piv] = A[piv], A[col]
            iv = pow(A[col][col], q - 2, q); A[col] = [v * iv % q for v in A[col]]
            for r in range(m):
                if r != col and A[r][col]:
                    f = A[r][col]; A[r] = [(vr - f * vc) % q for vr, vc in zip(A[r], A[col])]
        return np.array([[A[r][m + cc] for cc in range(m)] for r in range(m)], dtype=np.int64)
    best = -1e9; best_info = None; t0 = time.time()
    for j0 in range(0, n - s + 1, s):
        V = [[int(powers[i, j0 + k]) for i in range(s)] for k in range(s)]     # V[k][i] = r_i^{j0+k}
        Vinv = inv_mat(V)                                                        # c = Vinv @ t|_J
        C = (box % q) @ Vinv.T % q                                               # rows: c vectors (exact: < 2^49)
        # evaluate t = C @ powers mod q in chunks with object-free arithmetic: split powers into 12-bit halves
        hi = powers >> 12; lo = powers & 0xFFF
        T = ((C @ lo) % q + (((C @ hi) % q) << 12)) % q                          # C < 2^23, lo < 2^12 -> < 2^35*s ok; hi < 2^11
        v = logB(T)
        i = int(np.argmax(v))
        if v[i] > best: best = float(v[i]); best_info = (j0, box[i].tolist(), int(np.count_nonzero(T[i])))
    print(f"    max log2 B = {best:.1f}  (block start {best_info[0]}, t|_J = {best_info[1]}, nonzero coords of t = {best_info[2]})   [{time.time()-t0:.0f}s]")
    return best

if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "c"]
    if "a" in which: part_a()
    if "c" in which:
        part_c((0, 128, 1))
        part_c((4, 10, 19))
