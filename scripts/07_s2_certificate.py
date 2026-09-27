"""
holder_bound.py -- the UNCONDITIONAL character-sum bound (paper Sec. 6.4) and the s=2 certificate.

Part 1: Lambda(L) = sum_{a in Z_q} phi(a)^L with phi(a) = min(1, 64/|a|_c), exactly, for the block
        counts L = floor(256/s).  Holder gives  sum_{t in V_S^perp} B(t) <= Lambda(L)^s  for EVERY
        slot set S of size s (no structure), and with square roots  <= Lambda(L/2)^{s} for the
        rank-zero refinement.  Also prints the resulting per-s mass bound
        C(256,s) * 2^6 * Lambda^{6s} * 2^{-76.99 s}.
Part 2: s = 2 certificate.  For every pair S = {i1 < i2} of slots, characters are
        t_j = c1 z1^j + c2 z2^j (z_i = zeta^(2i+1)); two consecutive coordinates x = (t_j, t_{j+1})
        determine t, and the next block is C x with C = M^2, M the companion matrix of
        (X - z1)(X - z2).  Two-block Holder:  sum_{t != 0} B(t) <= sum_{x != 0} (Phi(x) Phi(Cx))^{64}
        where Phi(x) = phi(x1) phi(x2).  The sum over x with max|x_j| > 128 is at most 2^-54
        (tail bound in the paper); the rest is enumerated EXACTLY over x in [-128,128]^2 minus 0.
        Reports the maximum over all 32640 pairs of the certified value.
Run:  python scripts/07_s2_certificate.py [R] [2]    (R, the default, is the certified bound)
"""
import math, sys, time
import numpy as np
from params import q, n

zeta = 1753
def root(i): return pow(zeta, 2 * int(i) + 1, q)

def phi_vals(K=64.0):
    a = np.arange(q, dtype=np.int64)
    ac = np.minimum(a, q - a).astype(np.float64); ac[0] = 1.0
    return np.minimum(1.0, K / ac)

def part1():
    print("PART 1  Holder constants Lambda(L) = sum_a phi(a)^L  (phi = min(1,64/|a|))")
    ph = phi_vals()
    print(f"   {'s':>4s} {'L':>4s} {'Lambda(L)':>10s} {'log2':>7s} {'Lambda(L/2)':>11s} {'log2':>7s} {'mass bound exponent (s0=0)':>28s}")
    tot = 0.0
    for s in (2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 128, 200, 256):
        L = 256 // s
        lam = float((ph ** L).sum()); lam_h = float((ph ** max(L // 2, 1)).sum())
        logC = math.lgamma(257) / math.log(2) - math.lgamma(s + 1) / math.log(2) - math.lgamma(257 - s) / math.log(2)
        E = logC + 6 + 6 * s * math.log2(lam) - 76.99 * s
        tot += 2.0 ** E
        print(f"   {s:4d} {L:4d} {lam:10.3f} {math.log2(lam):7.3f} {lam_h:11.3f} {math.log2(lam_h):7.3f} {E:28.1f}")
    print(f"   sum over listed s of 2^E  ~ 2^{math.log2(tot):.1f}   (dominated by s=2; s=3 term ~ 2^-77)")

def part2(X=128, Lhalf=64):
    print(f"PART 2  s=2 certificate: max over all C(256,2)={256*255//2} slot pairs of sum_{{x in [-{X},{X}]^2 \\ 0}} (Phi(x)Phi(Cx))^{Lhalf}")
    t0 = time.time()
    r = np.arange(-X, X + 1, dtype=np.int64)
    x1, x2 = np.meshgrid(r, r, indexing='ij'); x1 = x1.ravel(); x2 = x2.ravel()
    keep = (x1 != 0) | (x2 != 0); x1 = x1[keep]; x2 = x2[keep]
    def phiL(v):
        vm = v % q; vc = np.minimum(vm, q - vm).astype(np.float64); vc = np.maximum(vc, 1.0)
        return np.minimum(1.0, 64.0 / vc) ** Lhalf
    base = phiL(x1) * phiL(x2)                       # Phi(x)^{L/2}
    roots = [root(i) for i in range(n)]
    worst = -1.0; worst_S = None; hist = {}
    cnt = 0
    for i1 in range(n):
        z1 = roots[i1]
        for i2 in range(i1 + 1, n):
            z2 = roots[i2]
            # companion M of X^2 - (z1+z2) X + z1 z2 acting on (t_j, t_{j+1}) -> (t_{j+1}, t_{j+2}); C = M^2
            p1 = (z1 + z2) % q; p0 = (-z1 * z2) % q            # t_{j+2} = p1 t_{j+1} + p0 t_j
            # M = [[0,1],[p0,p1]];  M^2 = [[p0, p1],[p0*p1, p0 + p1^2]]
            c00, c01 = p0, p1
            c10, c11 = (p0 * p1) % q, (p0 + p1 * p1) % q
            y1 = (c00 * x1 + c01 * x2) % q                       # exact: < 2^23 * 2^8 * 2 < 2^33
            y2 = (c10 * x1 + c11 * x2) % q
            val = float((base * phiL(y1) * phiL(y2)).sum())
            cnt += 1
            if val > worst: worst, worst_S = val, (i1, i2)
            b = int(math.floor(math.log2(max(val, 1e-300))))
            hist[b] = hist.get(b, 0) + 1
    print(f"   pairs done: {cnt}  [{time.time()-t0:.0f}s]")
    print(f"   worst certified value (before tail 2^-54): {worst:.3e} = 2^{math.log2(worst):.1f} at S={worst_S}")
    print("   histogram of log2(value) over pairs:", dict(sorted(hist.items())))
    return worst

def part_rigorous(X=128, Lhalf=64):
    """Certified version of Lemma "s = 2 certificate": no floating point enters the bound.

    For x in the box, the summand is (Phi(x) Phi(Cx))^64 = (2^24 / P(x))^64 with the INTEGER
    P(x) = prod over the four coordinates v of (x, Cx) of max(64, |v|_c).  So the box part of the
    sum is at most N * (2^24 / Pmin)^64, N = 257^2 - 1.  Pmin is found exactly: products of the four
    factors in float64 are within a relative 2^-51 of the true integer products (each factor is an
    integer < 2^23, hence exact, and IEEE multiplication is correctly rounded), so every x whose
    float product is within a factor 1 + 2^-40 of the float minimum is re-evaluated in exact integer
    arithmetic, and the true minimiser is among them.  The tail is bounded by the rational
    2 * tail_up(64) * Lambda_up(64) of rigorous.py.
    """
    from fractions import Fraction
    from rigorous import Lambda_up, tail_up, log2_up, fmt
    print(f"PART R  s=2 certificate, certified: all C(256,2)={256*255//2} slot pairs")
    t0 = time.time()
    r = np.arange(-X, X + 1, dtype=np.int64)
    x1, x2 = np.meshgrid(r, r, indexing='ij'); x1 = x1.ravel(); x2 = x2.ravel()
    keep = (x1 != 0) | (x2 != 0); x1 = x1[keep]; x2 = x2[keep]
    N = len(x1)
    def fac(v):                                        # max(64, |v|_c) as exact integers (int64)
        vm = v % q; vc = np.minimum(vm, q - vm)
        return np.maximum(vc, 64)
    fx = fac(x1).astype(np.float64) * fac(x2).astype(np.float64)
    roots = [root(i) for i in range(n)]
    Pmin_global = None; S_global = None; ncand_max = 0
    for i1 in range(n):
        z1 = roots[i1]
        for i2 in range(i1 + 1, n):
            z2 = roots[i2]
            p1 = (z1 + z2) % q; p0 = (-z1 * z2) % q
            c00, c01 = p0, p1
            c10, c11 = (p0 * p1) % q, (p0 + p1 * p1) % q
            y1 = (c00 * x1 + c01 * x2) % q
            y2 = (c10 * x1 + c11 * x2) % q
            Pf = fx * fac(y1).astype(np.float64) * fac(y2).astype(np.float64)
            fmin = Pf.min()
            cand = np.flatnonzero(Pf <= fmin * (1.0 + 2.0 ** -40))
            ncand_max = max(ncand_max, len(cand))
            Pmin = min(int(fac(np.array([x1[j]]))[0]) * int(fac(np.array([x2[j]]))[0])
                       * int(fac(np.array([y1[j]]))[0]) * int(fac(np.array([y2[j]]))[0]) for j in cand)
            if Pmin_global is None or Pmin < Pmin_global:
                Pmin_global, S_global = Pmin, (i1, i2)
    box = Fraction(N * 2 ** (24 * Lhalf), Pmin_global ** Lhalf)
    tail = 2 * tail_up(Lhalf) * Lambda_up(Lhalf)
    total = tail + box
    print(f"   pairs done  [{time.time()-t0:.0f}s];  largest candidate set {ncand_max}")
    print(f"   smallest P over all pairs and all x: {Pmin_global} at S={S_global}")
    print(f"   box part   <= N (2^24/Pmin)^64 <= {fmt(log2_up(box))}")
    print(f"   tail part  <= 2 tail_up(64) Lambda_up(64) <= {fmt(log2_up(tail))}"
          f"   [tail_up(64) <= {fmt(log2_up(tail_up(Lhalf)))}, Lambda_up(64) <= {fmt(log2_up(Lambda_up(Lhalf)))}]")
    print(f"   CERTIFIED: Sigma_S <= {fmt(log2_up(total))} for every two-element slot set S")
    return total

if __name__ == "__main__":
    which = sys.argv[1:] or ["R"]          # R: certified bound (used in the paper); 2: float sum, for comparison
    if "1" in which: part1()
    if "2" in which: part2()
    if "R" in which: part_rigorous()
