"""
Ms.py -- independent check of Table "coset constants" and the Appendix "Checkable data" maximisers.
Route differs from the repo: log2 G_s(x) = (1/2) sum over the WHOLE subgroup H of order 512/s (generated
by zeta^s, zeta = 1753) of log2 phi(h x), phi(a) = min(1, K/|a|_c), for every x in Z_q^*, K in {64, 8}.
(phi is even and H = -H, so this equals the product over 256/s coset representatives.)
Prints M_s, s*M_s, log2 eps_s = log2((1+(q-1)2^-M_s)^s - 1), the gap to the next-best orbit (which is
what makes the float argmax safe), and G_s at the paper's x_s computed with a = 1753^-s.
Run: python Ms.py [s ...]      (all s: about 4-5 min; s = 1 alone about 2.5 min)
"""
import numpy as np, math, time, sys
q = 8380417; zeta = 1753
v = np.arange(q, dtype=np.int64); vc = np.minimum(v, q - v).astype(np.float64); vc[0] = 1
LP = {K: -np.maximum(0.0, np.log2(vc / K)) for K in (64, 8)}
del vc
x = np.arange(1, q, dtype=np.int64)
XS = {1: 31705, 2: 8594, 4: 1089130, 8: 7646, 16: 89, 32: 89, 64: 49, 128: 61}
for s in ([int(a) for a in sys.argv[1:]] or [1, 2, 4, 8, 16, 32, 64, 128]):
    t0 = time.time(); g = pow(zeta, s, q); acc = {K: np.zeros(q - 1) for K in LP}; h = 1
    for _ in range(512 // s):
        idx = (x * h) % q
        for K in LP: acc[K] += LP[K][idx]
        h = h * g % q
    out = []
    for K in (64, 8):
        A = acc[K] / 2; j = int(np.argmax(A)); M = -A[j]
        rest = A[A < A[j] - 1e-6]; gap = A[j] - rest.max()
        u = math.log2(q - 1) - M
        eps = (math.log2(s) + u) if u < -40 else (s * math.log2(1 + 2.0 ** u) if u > 0 else math.log2((1 + 2.0 ** u) ** s - 1))
        out.append(f"K={K}: M_s={M:.4f} s*M_s={s*M:.2f} log2eps={eps:.3f} gap={gap:.3f}")
    a = pow(pow(zeta, s, q), q - 2, q); y = XS[s]; val = 0.0
    for _ in range(256 // s):
        val -= max(0.0, math.log2(min(y, q - y) / 64)); y = y * a % q
    print(f"s={s}: " + " | ".join(out) + f" | log2 G_s(x_s={XS[s]}) = {val:.4f} [{time.time()-t0:.0f}s]", flush=True)
