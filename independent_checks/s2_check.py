r"""
s2_check.py -- independent check of Lemma "s = 2 certificate", box part and tails.
(1) For all 32,640 two-slot sets S = {i1, i2} (omega_i = 1753^(2i+1) mod q) forms the block map
    C = V diag(w1^2, w2^2) V^-1, V = [[1,1],[w1,w2]], and for every x in [-128,128]^2 \ 0 checks
    P(x) = prod over the four coordinates v of (x, Cx) of max(64, |v|_c) >= T = 52,953,088 with the
    INTEGER test f(y1) f(y2) >= ceil(T / (f(x1) f(x2)))  -- no floating point in the decision --
    and lists the sets where P = T is attained.
(2) Upper bounds on tail(64) = sum_{|a|>128} phi^64 and Lambda(64) (phi = min(1,64/|a|)) by a 300-bit
    mpmath sum up to a = 20000 with explicit rounding slack plus the integral remainder; also the s=3
    parts (i) 3 tail(42) Lambda(42)^2 and (ii) 257^3 (64/513)^42.
Run: python s2_check.py      (about 1 min)
"""
import numpy as np, time
import mpmath as mp
q = 8380417; zeta = 1753; T = 52953088
mp.mp.prec = 300
def S(p, a0, K=20000, up=True):
    acc = mp.fsum(mp.power(mp.mpf(64) / a, p) for a in range(a0, K + 1))
    return acc * (1 + mp.mpf(2) ** -250) + mp.power(64, p) * mp.power(K, 1 - p) / (p - 1) if up else acc
lg = lambda v: float(mp.log(v, 2))
t64, L64 = 2 * S(64, 129), 129 + 2 * S(64, 65)
t42, L42 = 2 * S(42, 129), 129 + 2 * S(42, 65)
print(f"tail(64) <= 2^{lg(t64):.6f}  Lambda(64) <= 2^{lg(L64):.6f}  2 tail Lambda <= 2^{lg(2*t64*L64):.6f}")
print(f"box part (257^2-1)(2^24/T)^64 = 2^{lg((257**2-1)*(mp.mpf(2)**24/T)**64):.4f}")
print(f"s=3: (i) <= 2^{lg(3*t42*L42**2):.6f}   (ii) = 2^{lg(mp.mpf(257)**3*(mp.mpf(64)/513)**42):.4f}")
w = [pow(zeta, 2 * i + 1, q) for i in range(256)]
r = np.arange(-128, 129, dtype=np.int64)
X1, X2 = [a.ravel() for a in np.meshgrid(r, r, indexing='ij')]
nz = (X1 != 0) | (X2 != 0); X1, X2 = X1[nz], X2[nz]
f = lambda v: np.maximum(np.minimum(v % q, q - v % q), 64)
A = f(X1) * f(X2); need = (T + A - 1) // A
t0 = time.time(); below = 0; eq = []
for i1 in range(256):
    for i2 in range(i1 + 1, 256):
        a, b = w[i1], w[i2]; di = pow((b - a) % q, q - 2, q); a2, b2 = a * a % q, b * b % q
        c00 = (a2 * b - b2 * a) * di % q; c01 = (b2 - a2) * di % q
        c10 = (a * a2 * b - b * b2 * a) * di % q; c11 = (b * b2 - a * a2) * di % q
        B = f((c00 * X1 + c01 * X2) % q) * f((c10 * X1 + c11 * X2) % q)
        below += int((B < need).sum())
        k = int((A * B == T).sum())
        if k: eq.append((i1, i2, k))
print(f"x with P < T over all 32640 sets: {below}  ({'PASS' if below == 0 else 'FAIL'}); P = T attained at {eq}  [{time.time()-t0:.0f}s]")
