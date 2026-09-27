"""
mass_recheck.py -- corrected unconditional atypical-mass bound (paper Thm. "unconditional atypical
mass" and Thm. "unconditional bound"), in the form used by the paper.

Two points that the accounting has to get right:

(1) One-sided slots.  A dependent slot i of a pair (y, y') is of one of four types:
      Z  : y(zeta_i) = y'(zeta_i) = 0                      (rank zero)
      A  : y(zeta_i) = 0,  y'(zeta_i) != 0                  (rank one)
      B  : y'(zeta_i) = 0, y(zeta_i) != 0                   (rank one)
      P  : y'(zeta_i) = rho y(zeta_i), rho != 0, both != 0  (rank one)
    In the Poisson sum for a row weight W_r, t' is free on Z and on B (not only on Z), while t
    vanishes on B, so t ranges over the characters of S minus B (dimension s - |B|) and for each
    t the t' form a translate of the characters of Z u B (dimension |Z| + |B|).  On a collision
    F(u) = F(u') the weight 1/D'(F(u)) equals 1/D'(F(u')), so W_r(y,y') = W_r(y',y) and one may
    keep whichever side is better: with m := min(|A|, |B|) the character dimension is s - m and
    the number of free slots is d := |Z| + m.

(2) Conditioning on the z-data.  Y is y conditioned on the z-data l; it is a product of
    uniforms: 2 gamma1 - 2 beta - 1 = 1048183 values on an accepted coordinate, 2 beta + 1 = 393
    on a rejected one.  The Boudgoust-Lapiha window argument (any g cyclically consecutive
    coefficients of one polynomial determine its residue modulo the ideal of g slots) gives,
    choosing for each of the l polynomials the window with fewest rejected positions,
        Pr[ Y^(m) mod I_G = v ] <= 2^{-19.99946 g + 11.3811 floor(g r_m / 256)},
    r_m = number of rejected positions in polynomial m.  Over the l polynomials the loss is at
    most 11.3811 g r / 256 with r = total number of rejected positions.  A configuration with
    s dependent slots of which s0 rank-zero prescribes residues on g_y + g_y' = s + s0 slots
    of the pair, so its probability loses at most 11.3811 r (s + s0) / 256 bits, and averaging
    over l (r ~ Binomial(1280, 393 / 2^20), the law of l in the real experiment) multiplies it by
        F(c) = E[2^{c r} 1[r <= R]],   c = 11.3811 (s + s0) / 256,
    after discarding the z-data with more than R = 20 rejected positions, whose total probability
    Pr[r > R] is charged to the statistical distance directly.  The final bound is
        E_A SD <= Pr[r > R] + 1/2 sqrt( E_l[ (diag(l) + mass(l)) 1[r <= R] ] )      (Jensen).

Probabilities per slot (one nonce prescribed on one slot costs l * log2(1048183) = 99.9973 bits):
    Z: 2 x 99.9973     A, B: 99.9973     P: 99.9973 - log2(q - 1)
Weights: with s' := s - m the character dimension, per-row |W_r - 1| <= W(s', d), the minimum of
    Holder:   min_theta Lambda(theta L)^s' Lambda((1 - theta) L_d)^d,  L = floor(256/s'), L_d = floor(256/d)
    s' = 1:    q^d 2^-3598                         (Lemma "one dependent slot")
    s' = 2, 3: q^d Sigma, Sigma <= 2^-54.3, 2^-23.06   (the s = 2, 3 certificates)
and over six rows delta <= (1 + W)^6 - 1; independently delta <= min(q^{6(|Z|+m)} 2^7681, 2^22092)
(Lemma "counting bound", with the same symmetrisation).
Count: C(256, s) * s! / (|Z|! |A|! |B|! |P|!).

Run:  python scripts/06_atypical_mass.py        (a few minutes)
"""
import math, functools, numpy as np

q = 8380417; l = 5; gamma1 = 2**19; beta = 196; n = 256; k = 6
N_acc = 2 * gamma1 - 2 * beta - 1            # 1048183
N_rej = 2 * beta + 1                         # 393
p_rej = N_rej / (2 * gamma1)
b_acc = math.log2(N_acc)                     # 19.99946
loss = b_acc - math.log2(N_rej)              # 11.3811
slot = l * b_acc                             # 99.9973: one nonce, one slot
lq1 = math.log2(q - 1)
SUPP = 7681                                  # log2 |supp D'|
MAXW = 22092                                 # log2 max 1/D' over six rows

# ---- Lambda(p) = sum_a phi(a)^p exactly (phi(a) = min(1, 64/|a|_c))
lg = np.log(64.0 / np.arange(65, q // 2 + 1, dtype=np.float64))
_cache = {}
def L2(p):
    p = round(float(p), 9)
    if p not in _cache:
        _cache[p] = math.log2(q) if p == 0 else math.log2(129.0 + 2.0 * float(np.exp(p * lg).sum()))
    return _cache[p]

def lgC(a, b): return (math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)) / math.log(2)
LF = np.array([math.lgamma(i + 1) / math.log(2) for i in range(n + 1)])    # log2 i!

def delta6(w):
    """log2 of (1 + 2^w)^6 - 1, stable."""
    if w > 60: return 6 * w
    if w < -60: return math.log2(6) + w + 1e-15          # (1+x)^6 - 1 <= 6x(1+x)^5
    x = 2.0 ** w
    return math.log2(math.expm1(6 * math.log1p(x)))

R = 20
def lbinom(r): return lgC(1280, r) + r * math.log2(p_rej) + (1280 - r) * math.log2(1 - p_rej)
PR = np.array([lbinom(r) for r in range(R + 1)])                 # log2 Pr[r], r <= R
TAIL = math.log2(sum(2.0 ** lbinom(r) for r in range(R + 1, 200)))
def condF(c):
    """log2 E[2^{c r} 1[r <= R]]."""
    v = PR + c * np.arange(R + 1); m = v.max()
    return float(m + math.log2((2.0 ** (v - m)).sum()))

ths = np.linspace(0.0, 1.0, 201)
CERT = {2: -54.3, 3: -23.06}

@functools.lru_cache(maxsize=None)
def weight(s, d):
    """log2 bound on delta over six rows for a pair with s dependent slots and d free slots."""
    cands = []
    if s == 1:
        cands.append(d * math.log2(q) - 3598)
    else:
        L = n // s
        if d == 0:
            cands.append(s * L2(L))
        else:
            Ld = n // d
            cands.append(min(s * L2(t * L) + d * L2((1 - t) * Ld) for t in ths))
        if s in CERT: cands.append(d * math.log2(q) + CERT[s])
    w = delta6(min(cands))
    return min(w, 6 * d * math.log2(q) + SUPP, MAXW)

def run(sym=True, verbose=True):
    tot_mass = -np.inf; per_s = {}
    for s in range(1, n + 1):
        W = np.full((s // 2 + 1, s + 1), np.nan)
        for m in range(s // 2 + 1):
            for d in range(m, s - m + 1):                     # d = s0 + m with s0 + 2m <= s
                W[m, d] = weight(s - m, d)
        best = -1e18; acc = -np.inf
        for s0 in range(0, s + 1):
            rest = s - s0
            c = loss * (s + s0) / n
            cond = condF(c)                                             # E_l[2^{c r}; r <= R]
            for e in range(0, rest + 1):                                # e = |A| + |B|
                pP = rest - e
                base = (lgC(n, s) + LF[s] - LF[s0] - LF[pP]
                        - 2 * slot * s0 - slot * e - (slot - lq1) * pP + cond)
                ms = np.arange(0, e // 2 + 1)
                # multinomial part 1/(a! b!) over (a, b) with a + b = e, min(a, b) = m
                mult = -(LF[ms] + LF[e - ms]) + np.where(2 * ms == e, 0.0, 1.0)
                d = s0 + (ms if sym else e - ms)                         # unsym: worst side (B = larger)
                E = base + mult + W[ms, d]
                E = E[~np.isnan(E)]
                if E.size == 0: continue
                mx = float(E.max())
                acc = float(np.logaddexp2(acc, mx + math.log2((2.0 ** (E - mx)).sum())))
                best = max(best, mx)
        per_s[s] = (best, acc)
        tot_mass = float(np.logaddexp2(tot_mass, acc))
    # Lemma "diagonal": sum_y P(y)^2 (E_A[1/D'] - 1) <= max P(y) * max 1/D', averaged over r <= R
    diag = -1280 * b_acc + condF(loss) + MAXW
    return tot_mass, per_s, diag

if __name__ == "__main__":
    # Holder-only figures quoted in Theorem "unconditional atypical mass" (no certificates)
    saved = dict(CERT); CERT.clear(); weight.cache_clear()
    tot0, per0, _ = run(True)
    print(f"== Holder only: s=2 2^{per0[2][1]:.2f}  s=3 2^{per0[3][1]:.2f}  s=4 2^{per0[4][1]:.2f}  "
          f"s=5 2^{per0[5][1]:.2f}  total 2^{tot0:.2f}")
    CERT.update({2: saved[2]}); weight.cache_clear()
    tot1, _, _ = run(True)
    print(f"== s=2 certified only: total 2^{tot1:.2f}")
    CERT.update(saved); weight.cache_clear()
    print(f"per nonce-slot {slot:.4f} bits; rejected-coordinate loss {loss:.4f} bits; p_rej = {p_rej:.4e}")
    print(f"Lambda(128) = 2^{L2(128):.4f}, Lambda(64) = 2^{L2(64):.4f}, Lambda(1) = 2^{L2(1):.4f}")
    for sym in (True,):
        tot, per_s, diag = run(sym)
        tag = "symmetrised" if sym else "NOT symmetrised (t kept on the y side throughout)"
        print(f"\n== {tag}")
        for s in (1, 2, 3, 4, 5, 6, 8, 16, 64, 128, 129, 200, 256):
            print(f"   s={s:3d}  largest term 2^{per_s[s][0]:9.2f}   sum 2^{per_s[s][1]:9.2f}")
        worst_big = max(per_s[s][1] for s in range(129, n + 1))
        print(f"   sum over s >= 5: 2^{float(np.logaddexp2.reduce([per_s[s][1] for s in range(5, n + 1)])):.1f}")
        print(f"   largest per-s sum for s >= 129: 2^{worst_big:.1f}")
        print(f"   diagonal (averaged over the z-data): 2^{diag:.1f}")
        sd = 2.0 ** TAIL + 0.5 * math.sqrt(2.0 ** tot + 2.0 ** diag)
        print(f"   Pr[r > {R}] = 2^{TAIL:.1f}")
        print(f"   TOTAL atypical mass 2^{tot:.2f};   E_A SD <= Pr[r > R] + 1/2 sqrt(diag + mass) = 2^{math.log2(sd):.2f}")

# ---- Part 2: coset slot sets (Lemma "what the coset case proves"), with the same corrections.
# Weight per row: q^d * eps_s, eps_s = log2 of the last column of Table "coset constants";
# for S = a single slot eps_1 = 2^-3598.8.  Over six rows (1 + x)^6 - 1, capped by the counting bound.
EPS = {1: -3598.8, 2: -1758.8, 4: -850.1, 8: -397.2, 16: -166.4, 32: -50.8, 64: -5.74}
def coset_mass():
    out = {}
    for s, eps in EPS.items():
        acc = -np.inf
        for s0 in range(s + 1):
            for a in range(s - s0 + 1):
                for b in range(s - s0 - a + 1):
                    pP = s - s0 - a - b; m = min(a, b); d = s0 + m
                    c = loss * (s + s0) / n
                    w = min(delta6(d * math.log2(q) + eps), 6 * d * math.log2(q) + SUPP, MAXW)
                    E = (math.log2(n // s) + LF[s] - LF[s0] - LF[a] - LF[b] - LF[pP]
                         - 2 * slot * s0 - slot * (a + b) - (slot - lq1) * pP + condF(c) + w)
                    acc = float(np.logaddexp2(acc, E))
        out[s] = acc
    return out

if __name__ == "__main__":
    cm = coset_mass()
    print("\n== coset slot sets (single slot and cosets of order <= 64)")
    for s, v in cm.items(): print(f"   s={s:3d}  contribution 2^{v:.1f}")
    print(f"   total 2^{float(np.logaddexp2.reduce(list(cm.values()))):.1f}")
