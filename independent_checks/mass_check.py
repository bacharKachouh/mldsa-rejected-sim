"""
mass_check.py -- independent re-implementation of the atypical-mass sum
(Theorem "unconditional atypical mass" / Theorem "unconditional bound"). No repo imports.

For every s in 1..256 and every slot-kind split (s0, sA, sB, sP) it adds
  C(256,s) s!/(s0! sA! sB! sP!) (q-1)^sP E_r[2^{-(nu - lam r/256)(s+s0)} 1[r<=R]]
  * min((1+W)^6 - 1, q^{6d} 2^7681, 2^22092),
with m = min(sA,sB), d = s0+m, s' = s-m and a per-row weight W given by
  s' = 1        : q^d 2^-3598                       (single-slot lemma)
  s' >= 2       : min over theta in {k/GRID} of Lambda(theta L)^s' Lambda((1-theta) L0)^d,
                  L = floor(256/s'), L0 = floor(256/d)  (Hoelder; the second factor is 1 when d = 0)
  s' = 2 / 3    : also q^d 2^-54.3 / q^d 2^-23.06 when that certificate is switched on.

Lambda(p) = sum_{a in Z_q} phi(a)^p, computed by closed forms rather than a direct sum:
  phi = "64"      : min(1, 64/|a|_c)                    -> 129 + 2*64^p*(zeta(p,65) - zeta(p,H+1))
  phi = "refined" : min(1, 8/|a| + min(6289/q, 16/|a|)), phi(0) = 1   (the default)
      |a| <= 8              : phi = 1
      9 <= |a| <= A1        : phi = 8/|a| + 6289/q      (A1 = floor(16 q/6289) = 21321; direct fsum)
      A1 < |a| <= H         : phi = 24/|a|              -> 2*24^p*(zeta(p,A1+1) - zeta(p,H+1))
  (H = (q-1)/2.) The self-test compares Lambda against a direct math.fsum at a few values of p.
Conditioning uses the exact binomial law Bin(1280, 393/2^20) with cutoff R; Pr[r > R] is an exact rational.
Diagonal (Lemma "diagonal", averaged over the z-data): 2^{-1280 log2(2^20-393)} E[2^{lam r} 1[r<=R]] 2^22092.
Final bound: Pr[r>R] + 1/2 sqrt(mass + diag).

Run:   python mass_check.py                       (refined phi, R = 40, grid 200; about 1 min)
       python mass_check.py --phi 64 --R 20       (the cruder factor phi_64 with cut-off R = 20)
"""
import argparse, math, time
from fractions import Fraction
import numpy as np
import mpmath as mp

ap = argparse.ArgumentParser()
ap.add_argument("--phi", default="refined", choices=["refined", "64"])
ap.add_argument("--R", type=int, default=40)
ap.add_argument("--grid", type=int, default=200)
ap.add_argument("--certs", default="all", help="comma list of: none,2,23  (default all three)")
args = ap.parse_args()

mp.mp.dps = 30
t0 = time.time()
q = 8380417; n = 256; H = (q - 1) // 2
lq = math.log2(q); lq1 = math.log2(q - 1)
Nacc = 2**20 - 393
bacc = math.log2(Nacc); nu = 5 * bacc; lam = math.log2(Nacc / 393)
C0 = mp.mpf(6289) / q
A1 = (16 * q) // 6289                       # last a with 16/a >= 6289/q

def lam64(p):
    if p == 1: S = mp.harmonic(H) - mp.harmonic(64)
    else: S = mp.zeta(p, 65) - mp.zeta(p, H + 1)
    return 129 + 2 * mp.power(64, p) * S

_A = np.arange(9, A1 + 1, dtype=np.float64)
_LOGF = np.log(8.0 / _A + 6289.0 / q)
def lam_ref(p):
    # |a| <= 8: phi = 1;  9 <= |a| <= A1: phi = 8/|a| + 6289/q (direct fsum, 21313 terms);
    # |a| > A1: phi = 24/|a| (Hurwitz zeta, or harmonic numbers at p = 1)
    mid = math.fsum(np.exp(float(p) * _LOGF))
    if p == 1: S2 = mp.harmonic(H) - mp.harmonic(A1)
    else: S2 = mp.zeta(p, A1 + 1) - mp.zeta(p, H + 1)
    return 17 + 2 * mid + 2 * mp.power(24, p) * S2

LAM = lam_ref if args.phi == "refined" else lam64
_c = {}
def log2Lam(p):
    key = round(float(p), 12)
    if key not in _c:
        _c[key] = lq if p == 0 else float(mp.log(LAM(mp.mpf(p)), 2))
    return _c[key]

# self-test against a direct sum
a_all = np.arange(1, H + 1, dtype=np.float64)
phi_all = (np.minimum(1.0, 8 / a_all + np.minimum(6289 / q, 16 / a_all)) if args.phi == "refined"
           else np.minimum(1.0, 64 / a_all))
for p in (0.37, 1, 2, 42, 64, 128):
    direct = 1 + 2 * math.fsum(phi_all ** p)
    print(f"selftest Lambda({p}): closed form 2^{log2Lam(p):.9f}  direct fsum 2^{math.log2(direct):.9f}")
del a_all, phi_all

GRID = args.grid
Ls = sorted({n // k for k in range(1, n + 1)})
LL = {L: np.array([log2Lam(mp.mpf(k) * L / GRID) for k in range(GRID + 1)]) for L in Ls}
print(f"Lambda tables done [{time.time()-t0:.0f}s]", flush=True)
for L in (128, 85, 64, 42, 2, 1):
    print(f"   Lambda({L}) = 2^{LL[L][GRID]:.5f}")

def six(w):
    if w < -50: return math.log2(6) + w + 1e-12
    if w > 50: return 6 * w + 1e-12
    return math.log2(math.expm1(6 * math.log1p(2.0 ** w)))

def weights(certs):
    Wt = np.full((n + 1, n + 1), np.nan)
    for sp in range(1, n + 1):
        L = n // sp
        for d in range(0, sp + 1):
            if sp == 1: c = [d * lq - 3598]
            else:
                c = [sp * LL[L][GRID]] if d == 0 else [float(np.min(sp * LL[L] + d * LL[n // d][::-1]))]
                if sp == 2 and "2" in certs: c.append(d * lq - 54.3)
                if sp == 3 and "3" in certs: c.append(d * lq - 23.06)
            Wt[sp, d] = min(six(min(c)), 6 * d * lq + 7681, 22092)
    return Wt

R = args.R
pr = Fraction(393, 2**20)
PRq = [math.comb(1280, r) * pr**r * (1 - pr)**(1280 - r) for r in range(R + 1)]
tailF = 1 - sum(PRq)
PR = [mp.mpf(x.numerator) / x.denominator for x in PRq]
_cf = {}
def cond(k):
    if k not in _cf:
        c = mp.mpf(lam) * k / 256
        _cf[k] = float(mp.log(mp.fsum(PR[r] * mp.power(2, c * r) for r in range(R + 1)), 2))
    return _cf[k]
LF = np.array([math.lgamma(i + 1) / math.log(2) for i in range(n + 1)])
def lC(a, b): return LF[a] - LF[b] - LF[a - b]

def run(certs):
    Wt = weights(certs); per_s = {}
    for s in range(1, n + 1):
        terms = []
        for s0 in range(0, s + 1):
            cf = cond(s + s0)
            for e in range(0, s - s0 + 1):
                p = s - s0 - e
                base = lC(n, s) + LF[s] - LF[s0] - LF[p] + p * lq1 - nu * (s + s0) + cf
                a = np.arange(e + 1); m = np.minimum(a, e - a)
                terms.extend((base - LF[a] - LF[e - a] + Wt[s - m, s0 + m]).tolist())
        mx = max(terms)
        per_s[s] = mx + math.log2(math.fsum(2.0 ** (t - mx) for t in terms))
    v = list(per_s.values()); mx = max(v)
    return mx + math.log2(math.fsum(2.0 ** (x - mx) for x in v)), per_s

diag = -1280 * bacc + cond(256) + 22092          # cond(256) = log2 E[2^{lam r} 1[r<=R]]
tail = float(mp.log(mp.mpf(tailF.numerator) / tailF.denominator, 2))
print(f"phi={args.phi}  R={R}  grid=1/{GRID}")
print(f"Pr[r > {R}] = 2^{tail:.3f}   (exact rational; log2 = log2 num - log2 den = {tailF.numerator.bit_length() - tailF.denominator.bit_length()} +- 1)")
print(f"diagonal (averaged over z-data) = 2^{diag:.1f}")
sets = {"none": "", "2": "2", "23": "23"}
chosen = list(sets) if args.certs == "all" else args.certs.split(",")
for name in chosen:
    tot, ps = run(sets[name])
    sd = mp.mpf(2) ** tail + mp.sqrt(mp.mpf(2) ** tot + mp.mpf(2) ** diag) / 2
    ge5 = [ps[s] for s in ps if s >= 5]; m5 = max(ge5)
    print(f"certs={name:>4s}: total mass 2^{tot:.3f} | s2 2^{ps[2]:.2f} s3 2^{ps[3]:.2f} s4 2^{ps[4]:.2f} "
          f"s5 2^{ps[5]:.2f} | s>=5 2^{m5 + math.log2(math.fsum(2.0**(x-m5) for x in ge5)):.2f} "
          f"| max s>=129 2^{max(ps[s] for s in ps if s >= 129):.1f} | SD <= 2^{float(mp.log(sd, 2)):.3f}  [{time.time()-t0:.0f}s]",
          flush=True)
