"""
03b -- stress test of the atypical-pair bound (atypical-pair contributions to the
leftover-hash chi^2 for y -> HighBits(Ay)) before trying to prove it.

PASS 1  exact toy: per-pair collision weight  W(y,y') = E_A[ 1[F_A(y)=F_A(y')] / D'(F_A(y)) ]
        grouped by s = number of NTT slots where [y|y'] has rank < 2.  Lemma T needs
        sum_s P_s * (E[W|s] - 1) small; a pair with W >> 1 and non-negligible weight kills it.
PASS 2  ML-DSA-65: do small ring elements with MANY NTT zeros exist?  (If yes, structured pairs
        y' = y + f*e would make many-dependent-slot pairs far more likely than q^{-4s}.)
        Exhaustive over trinomials 1 +- X^b +- X^c ; random sample of ternary weight 4..8.
PASS 3  ML-DSA-65: character bound  B(t) = prod_j min(1, q/(2 alpha |t_j|_c))  for characters
        supported on s slots, s = 2..8: (a) structured slot sets (cosets {z, -z}, {z, iz, -z, -iz},
        ...) with the sparse choice of coefficients that zeroes 1 - 1/s of the coordinates,
        (b) random dense coefficients on random / structured slot sets.  Reports max log2 B.
Run:  python scripts/03b_toy_pair_weights.py [1] [2] [3]
"""
import itertools, math, sys, time
import numpy as np

# ============================================================================ PASS 1
def pass1(q, n, k, l, m, ybox, max_pairs_report=5):
    from fractions import Fraction
    alpha = (q - 1) // m; assert m * alpha + 1 == q and n == 2
    z = next(g for g in range(2, q) if pow(g, 2 * n, q) == 1 and pow(g, n, q) == q - 1)
    pts = [pow(z, 2 * i + 1, q) for i in range(n)]
    cells = m + 1
    pc = np.array([alpha / q] * m + [1 / q])
    coef = range(-ybox, ybox + 1)
    ys = np.array(list(itertools.product(itertools.product(coef, repeat=n), repeat=l)), dtype=np.int64)  # Y x l x n
    Y = len(ys)
    polys = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)               # q^n x n
    As = np.array(list(itertools.product(range(len(polys)), repeat=k * l)), dtype=np.int64)     # NA x (k*l)
    NA = len(As)
    print(f"PASS 1  q={q} n={n} k={k} l={l} m={m} |box|={Y} |A|={NA} |range|={cells**(k*n)}")
    # slot data for s-classification
    def slots(y):
        return [tuple(sum(int(y[c][j]) * pow(x, j, q) for j in range(n)) % q for c in range(l)) for x in pts]
    S = [slots(y) for y in ys]
    def dep(u, v):
        if all(x == 0 for x in u) or all(x == 0 for x in v): return True
        for a, b in zip(u, v):
            if a != 0:
                r = b * pow(a, q - 2, q) % q
                return all((r * a2 - b2) % q == 0 for a2, b2 in zip(u, v))
        return True
    sdeg = np.zeros((Y, Y), dtype=np.int8)
    for i in range(Y):
        for j in range(Y):
            if i != j: sdeg[i, j] = sum(dep(S[i][t], S[j][t]) for t in range(n))
    # F_A(y) for all A (chunked), n=2 negacyclic product explicit
    ymod = ys % q
    W = np.zeros((Y, Y))                      # accumulated collision weights
    chunk = 4096
    for start in range(0, NA, chunk):
        Ab = As[start:start + chunk]           # B x (k*l)
        B = len(Ab)
        out = np.zeros((B, Y, k * n), dtype=np.int64)
        for r in range(k):
            acc = np.zeros((B, Y, n), dtype=np.int64)
            for c in range(l):
                a = polys[Ab[:, r * l + c]]    # B x 2
                b = ymod[:, c, :]              # Y x 2
                a0, a1 = a[:, 0][:, None], a[:, 1][:, None]
                b0, b1 = b[:, 0][None, :], b[:, 1][None, :]
                p0 = (a0 * b0 - a1 * b1) % q
                p1 = (a0 * b1 + a1 * b0) % q
                acc[:, :, 0] = (acc[:, :, 0] + p0) % q; acc[:, :, 1] = (acc[:, :, 1] + p1) % q
            out[:, :, r * n:(r + 1) * n] = acc // alpha
        # encode cell vectors as ints, D' via cell counts
        code = np.zeros((B, Y), dtype=np.int64)
        logD = np.zeros((B, Y))
        for t in range(k * n):
            code = code * cells + out[:, :, t]
            logD += np.log(pc[out[:, :, t]])
        invD = np.exp(-logD)                   # 1/D'(F_A(y))
        for bi in range(B):
            cb = code[bi]
            order = np.argsort(cb, kind='stable'); cs = cb[order]
            bounds = np.flatnonzero(np.diff(cs)) + 1
            groups = np.split(order, bounds)
            for g in groups:
                if len(g) > 1:
                    w = invD[bi, g[0]]
                    W[np.ix_(g, g)] += w
    W /= NA
    np.fill_diagonal(W, 0)
    tot_pairs = Y * (Y - 1)
    print(f"   s   #ordered pairs   P_s        mean W-1        max W-1      contribution P_s*(mean W-1)")
    total = 0.0
    for s in range(n + 1):
        mask = (sdeg == s); np.fill_diagonal(mask, False)
        cnt = int(mask.sum())
        if cnt == 0: continue
        Ws = W[mask]; Ps = cnt / (Y * Y)
        print(f"   {s}   {cnt:9d}      {Ps:.3e}   {Ws.mean()-1:+.4e}   {Ws.max()-1:+.4e}   {Ps*(Ws.mean()-1):+.3e}")
        total += Ps * (Ws.mean() - 1)
    lead = (1 / Y) * (cells ** (k * n) - 1)
    print(f"   sum over s (atypical + typical) = {total:+.3e};  min-entropy term = {lead:.3e};  chi2 = {lead + total:.3e}")
    # worst pairs
    idx = np.dstack(np.unravel_index(np.argsort(-W, axis=None)[:max_pairs_report], W.shape))[0]
    for i, j in idx:
        print(f"   worst pair: s={sdeg[i,j]}  W-1={W[i,j]-1:+.3e}   y={ys[i].tolist()}  y'={ys[j].tolist()}")

# ============================================================================ PASS 2
def pass2():
    from params import q, n
    zeta = 1753
    roots = np.array([pow(zeta, 2 * i + 1, q) for i in range(n)], dtype=np.int64)
    P = np.zeros((n, n), dtype=np.int64)       # P[i,j] = roots[i]^j
    acc = np.ones(n, dtype=np.int64)
    for j in range(n):
        P[:, j] = acc; acc = acc * roots % q
    print("PASS 2  small ring elements with many NTT zeros (ML-DSA-65, q=8380417, n=256)")
    # trinomials 1 + s1 X^b + s2 X^c, 0<b<c<256 (rotations X^a * f have the same zero set)
    best = 0; bestf = None; t = time.time()
    for s1 in (1, -1):
        for s2 in (1, -1):
            for b in range(1, n):
                # vectorize over c
                cs = np.arange(b + 1, n)
                if len(cs) == 0: continue
                vals = (1 + s1 * P[:, b][:, None] + s2 * P[:, cs]) % q   # n x len(cs)
                zeros = (vals == 0).sum(axis=0)
                i = int(np.argmax(zeros))
                if zeros[i] > best: best = int(zeros[i]); bestf = (s1, b, s2, int(cs[i]))
    desc = "none has a zero" if bestf is None else f"f = 1 {'+' if bestf[0]>0 else '-'} X^{bestf[1]} {'+' if bestf[2]>0 else '-'} X^{bestf[3]}"
    print(f"   all {4*n*(n-1)//2} trinomials 1+-X^b+-X^c: max #NTT zeros = {best}  ({desc})  [{time.time()-t:.0f}s]")
    rng = np.random.default_rng(1)
    for wgt in (4, 5, 6, 8):
        best = 0; N = 40000
        idx = np.array([rng.choice(n, wgt, replace=False) for _ in range(N)])
        sgn = rng.choice([-1, 1], size=(N, wgt))
        # evaluate: f(zeta_i) = sum_t sgn_t * roots_i^{idx_t}
        vals = np.zeros((N, n), dtype=np.int64)
        for tpos in range(wgt):
            vals = (vals + sgn[:, tpos][:, None] * P[:, idx[:, tpos]].T) % q
        zeros = (vals == 0).sum(axis=1)
        print(f"   random ternary weight {wgt} ({N} samples): max #NTT zeros = {int(zeros.max())}, mean = {zeros.mean():.4f}  (uniform-slot heuristic mean = 256/q = {256/q:.2e})")
    print("   => the heuristic P_s ~ q^{-4s} for s dependent slots is NOT undermined by structured small differences"
          " if the maxima above stay O(1).")

# ============================================================================ PASS 3
def pass3():
    from params import q, n, alpha
    zeta = 1753
    roots = [pow(zeta, 2 * i + 1, q) for i in range(n)]
    # time-domain basis vector for slot i: e_i[j] ~ roots[i]^{-j}  (up to scaling); a character
    # supported on slots S has t_j = sum_{i in S} c_i * roots[i]^{-j}.
    inv = [pow(r, q - 2, q) for r in roots]
    E = np.zeros((n, n), dtype=np.int64)       # E[i, j] = roots[i]^{-j}
    for i in range(n):
        acc = 1
        for j in range(n):
            E[i, j] = acc; acc = acc * inv[i] % q
    def logB(t):
        tc = np.minimum(t % q, q - (t % q)); tc = np.maximum(tc, 1)
        g = np.minimum(1.0, (q / (2.0 * alpha)) / tc)
        return float(np.log2(g).sum())
    print("PASS 3  character bound log2 B(t) = sum_j log2 min(1, q/(2 alpha |t_j|)) for s-slot characters (ML-DSA-65)")
    print(f"   reference: random dense character ~ {256*math.log2(q/(2*alpha)) - 256*(math.log2(q)-1):.0f} (heuristic); s=1 exact max = -4389.8")
    # slot index arithmetic: roots[i] = zeta^{2i+1}; multiplying by zeta^{2d} maps slot i -> i+d; -1 = zeta^256 -> shift by 128; i = zeta^128 -> shift 64
    rng = np.random.default_rng(3)
    for s, shifts in ((2, [0, 128]), (4, [0, 64, 128, 192]), (8, [0, 32, 64, 96, 128, 160, 192, 224]), (16, list(range(0, 256, 16)))):
        # structured coset: slots {i0 + d}; sparse coefficient choice: c = DFT-delta so that t vanishes on all but 1/s of coords
        i0 = 5
        S = [(i0 + d) % n for d in shifts]
        best_sparse = -1e9; best_dense = -1e9
        for trial in range(200):
            # sparse: t_j = c * roots[i0]^{-j} * (1/s) sum_d omega^{-dj} ... construct directly: choose residue class r0 and scale c
            c = int(rng.integers(1, q)); r0 = int(rng.integers(0, s))
            t = np.zeros(n, dtype=np.int64)
            js = np.arange(r0, n, s)
            t[js] = (c * E[i0, js]) % q
            best_sparse = max(best_sparse, logB(t))
            # dense: random coefficients on the s slots
            cs = rng.integers(1, q, size=s)
            t = np.zeros(n, dtype=np.int64)
            for ci, si in zip(cs, S): t = (t + int(ci) * E[si]) % q
            best_dense = max(best_dense, logB(t))
        # random slot set, dense
        best_rand = -1e9
        for trial in range(200):
            S2 = rng.choice(n, s, replace=False); cs = rng.integers(1, q, size=s)
            t = np.zeros(n, dtype=np.int64)
            for ci, si in zip(cs, S2): t = (t + int(ci) * E[si]) % q
            best_rand = max(best_rand, logB(t))
        # summed over the q^s characters per row (crude): log2 count = s*23; requirement for negligibility vs P_s: log2 B + 6*s*23 + 6*(-92 s) ... report raw
        print(f"   s={s:2d}: coset slot set, sparse coeffs (t on 1/{s} of coords): max log2 B = {best_sparse:8.1f};  dense coeffs: {best_dense:8.1f};  random slots dense: {best_rand:8.1f}"
              f"   | crude 6-row sum bound 2^({6*s*23 + 6*best_sparse:.0f}) vs weight P_s = 2^-{92*s}")
    print("   => Lemma T at s is safe if  6*(23 s + max log2 B)  <  92 s - 100 ; the sparse structured characters are the worst found.")

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which:
        pass1(q=13, n=2, k=1, l=2, m=4, ybox=2)
        pass1(q=17, n=2, k=1, l=2, m=4, ybox=2)
    if "2" in which: pass2()
    if "3" in which: pass3()
