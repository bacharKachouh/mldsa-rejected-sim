"""
03a -- EXACT toy-scale test of the leftover-hash argument (paper, collision form):
is the family  F_A : y -> HighBits(A y)   (A uniform over R_q^{k x l}, y uniform on a box)
"universal relative to D = law of HighBits(U), U uniform", so that
      E_A[ chi^2( P_A || D ) ]  =  sum_{y,y'} P(y)P(y') E_A[ 1[F_A(y)=F_A(y')] / D(F_A(y)) ] - 1
is ~ P[y = y'] * |supp D|  (the min-entropy term) plus a negligible atypical-pair term?

Toy parameters mimic ML-DSA's structure: R_q = Z_q[X]/(X^n+1) with q = 1 mod 2n (fully
splitting NTT), q = m*alpha + 1 with m cells, HighBits(r) = floor(r/alpha) for r in [0,q)
(clean form; the FIPS relabeling is a bijection and does not change the argument).
Everything below is exact rational arithmetic over full enumeration of A and y.

For each parameter set we report:
  H   = log2 |box|              (min-entropy of y)
  m_out = log2 |range|          (output bits)
  chi2 = E_A[chi^2(P_A || D)]   (exact)
  predicted lower-order term  P[y=y'] * (E_y[1/D(F(y))] - 1)
  E_A[SD(P_A, D)]               (exact average statistical distance)
  bound (1/2)sqrt(chi2)
and the fraction of y-pairs that are 'atypical' (some NTT slot with dependent (y(z), y'(z))).
"""
import itertools, math, sys
from fractions import Fraction

def primitive_2n_root(q, n):
    for g in range(2, q):
        if pow(g, 2 * n, q) == 1 and pow(g, n, q) == q - 1:
            return g
    raise ValueError

def ntt_points(q, n):
    z = primitive_2n_root(q, n)
    return [pow(z, 2 * i + 1, q) for i in range(n)]

def evalp(poly, x, q):
    return sum(c * pow(x, j, q) for j, c in enumerate(poly)) % q

def polymul(a, b, q, n):
    r = [0] * (2 * n)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            r[i + j] += ai * bj
    return [(r[i] - r[i + n]) % q for i in range(n)]

def run(q, n, k, l, ybound, verbose=True):
    alpha = None
    for m in range(2, q):
        if (q - 1) % m == 0 and (q - 1) // m > 1:
            pass
    # choose alpha as (q-1)/m with m = number of cells given; caller passes q with q-1 = m*alpha
    return

def experiment(q, n, k, l, m, ybox):
    """q = m*alpha+1; box = [-ybox, ybox]^{l*n}; enumerate all A and all y."""
    alpha = (q - 1) // m
    assert m * alpha + 1 == q
    pts = ntt_points(q, n)
    cells = m + 1  # HighBits in {0..m} (value q-1 -> m, the wrap cell of size 1: clean form)
    def hb(r): return r // alpha
    # target law D over cell vectors: product over k*n coordinates of P[hb(U)=c]
    pc = [Fraction(alpha, q)] * m + [Fraction(1, q)]   # cell m has the single value q-1... adjust:
    # clean form: r in [0,q): floor(r/alpha) in {0..m}; cell c<m has alpha values, cell m has 1 value (r=q-1)
    def D_of(cellvec):
        p = Fraction(1)
        for c in cellvec: p *= pc[c]
        return p
    # enumerate y (all polys of l coordinates with coeffs in [-ybox, ybox])
    coef_range = list(range(-ybox, ybox + 1))
    ys = list(itertools.product(itertools.product(coef_range, repeat=n), repeat=l))
    Y = len(ys)
    # enumerate A: k*l polynomials with coeffs in Z_q
    polys = list(itertools.product(range(q), repeat=n))
    As = list(itertools.product(polys, repeat=k * l))
    NA = len(As)
    print(f"  q={q} n={n} k={k} l={l} m={m} alpha={alpha} |box|={Y} (H={math.log2(Y):.2f} bits) "
          f"|range|={cells**(k*n)} (m_out={math.log2(cells**(k*n)):.2f} bits) |A|={NA}")
    # F_A(y) for all A, y
    def F(A, y):
        out = []
        for r in range(k):
            acc = [0] * n
            for c in range(l):
                p = polymul(list(A[r * l + c]), [v % q for v in y[c]], q, n)
                acc = [(a + b) % q for a, b in zip(acc, p)]
            out.extend(hb(v) for v in acc)
        return tuple(out)
    chi2_sum = Fraction(0); sd_sum = Fraction(0)
    for A in As:
        counts = {}
        for y in ys:
            w = F(A, y); counts[w] = counts.get(w, 0) + 1
        chi2 = sum(Fraction(cnt, Y) ** 2 / D_of(w) for w, cnt in counts.items()) - 1
        # SD(P_A, D): sum over all cell vectors; D on vectors not hit contributes D(w)
        hit = sum(D_of(w) for w in counts)
        sd = Fraction(1, 2) * (sum(abs(Fraction(cnt, Y) - D_of(w)) for w, cnt in counts.items()) + (1 - hit))
        chi2_sum += chi2; sd_sum += sd
    chi2_avg = chi2_sum / NA; sd_avg = sd_sum / NA
    # predicted leading term: P[y=y'] * (E_y[1/D(F(y))] - 1); for typical y, F_A(y) ~ D so E[1/D] = |supp| = cells^{kn}
    lead = Fraction(1, Y) * (Fraction(cells ** (k * n)) - 1)
    # atypical pair fraction: some slot where (y(z), y'(z)) in Z_q^l are dependent
    def slots(y): return [tuple(evalp([v % q for v in y[c]], x, q) for c in range(l)) for x in pts]
    S = [slots(y) for y in ys]
    def dep(u, v):
        # rank < 2 in Z_q^l: u == 0 or v == 0 or v = r u for some r
        if all(x == 0 for x in u) or all(x == 0 for x in v): return True
        # find r from first nonzero coordinate of u
        for a, b in zip(u, v):
            if a != 0:
                r = b * pow(a, q - 2, q) % q
                return all((r * a2 - b2) % q == 0 for a2, b2 in zip(u, v))
        return True
    atyp = 0
    for i in range(Y):
        for j in range(Y):
            if i != j and any(dep(S[i][s], S[j][s]) for s in range(n)): atyp += 1
    print(f"    E_A[chi2(P_A||D)] = {float(chi2_avg):.3e}   predicted min-entropy term P[y=y'](|range|-1) = {float(lead):.3e}"
          f"   ratio = {float(chi2_avg / lead):.4f}")
    print(f"    E_A[SD(P_A, D)]   = {float(sd_avg):.3e}   bound (1/2)sqrt(chi2) = {0.5 * math.sqrt(float(chi2_avg)):.3e}"
          f"   2^-(H-m_out)/2 = {2 ** (-(math.log2(Y) - math.log2(cells ** (k * n))) / 2):.3e}")
    print(f"    atypical (slot-dependent) ordered pairs: {atyp}/{Y * (Y - 1)} = {atyp / (Y * (Y - 1)):.3e}")
    return float(chi2_avg), float(lead), float(sd_avg)

if __name__ == "__main__":
    print("EXACT toy leftover-hash test for y -> HighBits(Ay)  (toy scale)")
    # n=2: q = 1 mod 4 and q = m*alpha+1.  q=17: m=4,alpha=4 or m=16,alpha=1 ; q=13: m=4, alpha=3; q=41: m=8, alpha=5; q=29: m=4, alpha=7
    # keep enumeration sizes manageable: |A| = q^(k*l*n), |box| = (2b+1)^(l*n)
    experiment(q=13, n=2, k=1, l=1, m=4, ybox=1)      # H=3.2 bits,  out 4.6 bits: no slack (sanity: large SD)
    experiment(q=13, n=2, k=1, l=1, m=4, ybox=3)      # H=5.6 bits,  out 4.6
    experiment(q=13, n=2, k=2, l=1, m=4, ybox=4)      # H=6.3 bits,  out 9.3: negative slack
    experiment(q=13, n=2, k=1, l=2, m=4, ybox=2)      # H=9.3 bits,  out 4.6: slack 4.7
    # experiment(q=17, n=2, k=1, l=2, m=4, ybox=3)   # H=11.2 bits: 2e8 evaluations, too slow in pure Python
