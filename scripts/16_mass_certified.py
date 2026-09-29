"""
16 -- the atypical mass with directed rounding (Theorem "unconditional atypical mass",
Theorem "unconditional bound", Theorem "existential unforgeability with revealed attempts").

Same bound as 06_atypical_mass.py and independent_checks/mass_check.py, evaluated so that every
printed value is a certified UPPER bound; no step relies on an estimate of floating-point error.

  * Transcendental leaves (log2 q, nu, lambda, log2 i!, the binomial conditioning factor, the
    six-row map w -> log2((1+2^w)^6 - 1), exponentials in the final sum) are evaluated in mpmath
    interval arithmetic, and the endpoint that the bound needs is converted to a float rounded
    outward (up for quantities that enter with a plus sign, down for the others).
  * Lambda(p) = sum_{a in Z_q} phi(a)^p, refined phi = min(1, 8/|a|_c + min(6289/q, 16/|a|_c)), is
    bounded above for every p on the theta grid: |a| <= 8 contributes 17 exactly; 9 <= a <= 64
    term by term in intervals; 65 <= a <= A1 = 21321 in blocks [b, b') on which the decreasing
    phi^p is at most its value at b; a > A1, where phi = 24/a, by 24^p (a0^-p + int_{a0}^{H} x^-p dx).
  * The sum over (s, s0, e, a) is formed in float64 with every addition and multiplication
    followed by np.nextafter towards +inf (or -inf for subtracted quantities). IEEE arithmetic
    rounds to nearest, so the result bounds the exact value of the expression.
  * Terms whose certified log2 is above -400 are added in interval arithmetic; the others are
    bounded by (their number) * 2^(their maximum).

Run:  python scripts/16_mass_certified.py        (a few minutes)
"""
import math, time
from fractions import Fraction
import numpy as np
import mpmath as mp
from mpmath import iv

iv.prec = 96
t0 = time.time()
q = 8380417; n = 256; H = (q - 1) // 2; l = 5
Nacc = 2 ** 20 - 393; Nrej = 393
A1 = (16 * q) // 6289                       # last a with 16/a >= 6289/q
R = 40
INF = np.inf


def up(x):
    """float >= the upper endpoint of the interval (or number) x."""
    b = x.b if hasattr(x, "b") else mp.mpf(x)
    f = float(b)
    while mp.mpf(f) < b:
        f = math.nextafter(f, INF)
    return f


def dn(x):
    a = x.a if hasattr(x, "a") else mp.mpf(x)
    f = float(a)
    while mp.mpf(f) > a:
        f = math.nextafter(f, -INF)
    return f


def log2i(x):
    return iv.log(x) / iv.log(2)


# ---- leaves
LQ_up = up(log2i(iv.mpf(q)))
LQ1_up = up(log2i(iv.mpf(q - 1)))
NU_dn = dn(5 * log2i(iv.mpf(Nacc)))
LAM_iv = log2i(iv.mpf(Nacc)) - log2i(iv.mpf(Nrej))
LF_up = np.array([up(log2i(iv.mpf(math.factorial(i)))) for i in range(n + 1)])
LF_dn = np.array([dn(log2i(iv.mpf(math.factorial(i)))) for i in range(n + 1)])
LC_up = np.array([up(log2i(iv.mpf(math.comb(n, s)))) for s in range(n + 1)])
MAXW = 22092
assert up(1536 * (LQ_up - dn(log2i(iv.mpf(Nrej))))) <= MAXW     # max 1/D' over six rows
SUPP = 7681                                                     # |supp D'| <= 16^1536 * 2 * 2^1536

# binomial law of r, exact
pr = Fraction(Nrej, 2 ** 20)
PR = [Fraction(math.comb(1280, r)) * pr ** r * (1 - pr) ** (1280 - r) for r in range(R + 1)]
TAIL = 1 - sum(PR)                                              # Pr[r > R], exact rational
PR_iv = [iv.mpf(x.numerator) / x.denominator for x in PR]
COND_up = {}
def cond_up(k):
    """upper bound on log2 E[2^{lam k r / 256} 1[r <= R]]."""
    if k not in COND_up:
        c = LAM_iv * k / 256
        COND_up[k] = up(log2i(sum(PR_iv[r] * iv.exp(c * r * iv.log(2)) for r in range(R + 1))))
    return COND_up[k]


# ---- Lambda(p), certified upper bound on log2
C0 = iv.mpf(6289) / q
BL = [65]
while BL[-1] <= A1:
    BL.append(min(A1 + 1, max(BL[-1] + 1, int(BL[-1] * 1.02))))
LOGF_small = [iv.log(iv.mpf(8) / a + C0) for a in range(9, 65)]
LOGF_blk = [(iv.log(iv.mpf(8) / b + C0), b2 - b) for b, b2 in zip(BL[:-1], BL[1:])]
a0 = A1 + 1

def log2Lam_up(p):
    if p == 0:
        return LQ_up
    P = iv.mpf(p.numerator) / p.denominator if isinstance(p, Fraction) else iv.mpf(p)
    s = iv.mpf(17)
    s += 2 * sum(iv.exp(P * lf) for lf in LOGF_small)
    s += 2 * sum(cnt * iv.exp(P * lf) for lf, cnt in LOGF_blk)
    if p == 1:
        integ = iv.log(iv.mpf(H)) - iv.log(iv.mpf(a0))
    else:
        integ = (iv.exp((1 - P) * iv.log(iv.mpf(H))) - iv.exp((1 - P) * iv.log(iv.mpf(a0)))) / (1 - P)
    tail = iv.exp(P * iv.log(iv.mpf(24))) * (iv.exp(-P * iv.log(iv.mpf(a0))) + integ)
    return up(log2i(s + 2 * tail))


GRID = 200
Ls = sorted({n // k for k in range(1, n + 1)})
LL = {L: np.array([log2Lam_up(Fraction(k * L, GRID)) for k in range(GRID + 1)]) for L in Ls}
print(f"Lambda tables ({len(Ls)} values of L, grid 1/{GRID}) [{time.time()-t0:.0f}s]", flush=True)
for L in (128, 85, 64, 42, 2, 1):
    print(f"   log2 Lambda({L}) <= {LL[L][GRID]:.6f}")


def six_up(w):
    """upper bound on log2((1 + 2^w)^6 - 1) for a float w (itself an upper bound; monotone)."""
    x = iv.exp(iv.mpf(w) * iv.log(2))
    # expanded, so that no cancellation widens the interval when x is tiny
    return up(log2i(6 * x + 15 * x ** 2 + 20 * x ** 3 + 15 * x ** 4 + 6 * x ** 5 + x ** 6))


def au(a, b): return np.nextafter(np.add(a, b), INF)
def mu(a, b): return np.nextafter(np.multiply(a, b), INF)
def md(a, b): return np.nextafter(np.multiply(a, b), -INF)


def weights(certs):
    Wt = np.full((n + 1, n + 1), np.nan)
    for sp in range(1, n + 1):
        L = n // sp
        for d in range(0, sp + 1):
            if sp == 1:
                c = [float(au(mu(d, LQ_up), -3598.0))]
            else:
                if d == 0:
                    c = [float(mu(sp, LL[L][GRID]))]
                else:
                    c = [float(np.min(au(mu(sp, LL[L]), mu(d, LL[n // d][::-1]))))]
                if sp == 2 and 2 in certs: c.append(float(au(mu(d, LQ_up), -54.3)))
                if sp == 3 and 3 in certs: c.append(float(au(mu(d, LQ_up), -23.06)))
            Wt[sp, d] = min(six_up(min(c)), float(au(mu(6 * d, LQ_up), SUPP)), MAXW)
    return Wt


KEEP = -400.0

def run(certs):
    Wt = weights(certs)
    kept = []; rest_n = 0; rest_max = -INF; per_s_max = {}
    for s in range(1, n + 1):
        blocks = []
        for s0 in range(0, s + 1):
            cf = cond_up(s + s0)
            for e in range(0, s - s0 + 1):
                p = s - s0 - e
                # log2 of C(256,s) s!/(s0! p!) (q-1)^p 2^{-nu (s+s0)} E[...]
                base = au(au(au(au(LC_up[s], LF_up[s]), -LF_dn[s0]), -LF_dn[p]), mu(p, LQ1_up))
                base = au(au(base, -md(NU_dn, s + s0)), cf)
                a = np.arange(e + 1); m = np.minimum(a, e - a)
                T = au(au(au(base, -LF_dn[a]), -LF_dn[e - a]), Wt[s - m, s0 + m])
                blocks.append(T)
        T = np.concatenate(blocks)
        per_s_max[s] = float(T.max())
        big = T > KEEP
        kept.append((s, T[big]))
        if (~big).any():
            rest_n += int((~big).sum()); rest_max = max(rest_max, float(T[~big].max()))
    per_s = {}
    total = iv.mpf(0)
    for s, Ts in kept:
        ps = sum((iv.exp(iv.mpf(float(t)) * iv.log(2)) for t in Ts), iv.mpf(0))
        per_s[s] = ps; total += ps
    if rest_n:
        total += rest_n * iv.exp(iv.mpf(rest_max) * iv.log(2))
    return total, per_s, rest_n, rest_max


def fmt(x):
    return "2^" + (f"{up(log2i(x)):.4f}" if x.b > 0 else "-inf")


diag = iv.exp((-1280 * log2i(iv.mpf(Nacc)) + cond_up(256) + MAXW) * iv.log(2))
tail = iv.mpf(TAIL.numerator) / TAIL.denominator
print(f"Pr[r > {R}] <= {fmt(tail)}   diagonal (averaged over the z-data) <= {fmt(diag)}")
res = {}
for name, certs in (("none", ()), ("2", (2,)), ("2,3", (2, 3))):
    tot, per_s, rn, rm = run(certs)
    sd = tail + iv.sqrt(tot + diag) / 2
    res[name] = (tot, sd)
    print(f"certs={name:>4s}: mass <= {fmt(tot)} | s2 {fmt(per_s[2])} s3 {fmt(per_s[3])} s4 {fmt(per_s[4])}"
          f" s5 {fmt(per_s[5])} | {rn} terms below 2^{KEEP:.0f} (max 2^{rm:.1f}) | E_A SD <= {fmt(sd)}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

# ---- checks of the constants stated in the paper
two = lambda k: iv.exp(iv.mpf(k) * iv.log(2))
mass, sd = res["2,3"]; mass2, sd2 = res["2"]
EX = mass + two(-3051)                       # E_A[X(A)] <= diag bound + mass
EX2 = mass2 + two(-3051)
QA = two(67)
chk = [
    ("Theorem unconditional bound: M <= 2^-182.41", mass.b <= two(-182.41).a),
    ("Theorem unconditional bound: E_A SD < 2^-92.2", sd.b < two(-92.2).a),
    ("Remark two-slot: M <= 2^-135.9", mass2.b <= two(-135.9).a),
    ("Remark two-slot: E_A SD < 2^-68.9", sd2.b < two(-68.9).a),
    ("E_A X(A) < 2^-182.4", EX.b < two(-182.4).a),
]
markov = QA * iv.mpf(8.5) * EX + QA * tail
cross = iv.sqrt(2 * (iv.e - 1) * QA * EX)
markov2 = QA * iv.mpf(8.5) * EX2 + QA * tail
cross2 = iv.sqrt(2 * (iv.e - 1) * QA * EX2)
print(f"Theorem uf, Q_A = 2^67: additive Markov terms Q_A (6.5 + 2) E X + Q_A Pr[r>40] <= {fmt(markov)};"
      f" cross-term factor sqrt(2(e-1) Q_A E X) <= {fmt(cross)}")
print(f"   with the two-slot certificate only: additive <= {fmt(markov2)}; cross-term factor <= {fmt(cross2)}")
chk += [("additive terms < 2^-112.3", (markov + two(-189) + two(-257) + two(-362)).b < two(-112.3).a),
        ("cross-term factor < 2^-56.8", cross.b < two(-56.8).a),
        ("two-slot: additive < 2^-65.7", (markov2 + two(-189) + two(-257) + two(-362)).b < two(-65.7).a)]
for kappa in (20, 40, 60):
    life = QA * iv.mpf(6.5) * EX * two(kappa) + QA * tail
    print(f"   per key, all but a 2^-{kappa} fraction of matrices: lifetime additive loss <= {fmt(life)};"
          f" cross-term factor <= {fmt(iv.sqrt(2 * (iv.e - 1) * QA * EX * two(kappa)))}")
for label, ok in chk:
    print(f"   {'PASS' if ok else 'FAIL'}  {label}")
print(f"[{time.time()-t0:.0f}s]")
