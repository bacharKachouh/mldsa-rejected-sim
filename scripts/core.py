import numpy as np, hashlib
n, q = 256, 8380417
k, l, eta = 6, 5, 4
gamma1 = 1 << 19
gamma2 = (q - 1) // 32
alpha = 2 * gamma2
tau, beta, d, omega = 49, 196, 13, 55

def centered(x, m=q):
    r = np.asarray(x) % m
    return np.where(r > m // 2, r - m, r)

def zmul(a, b):
    """exact integer negacyclic product in Z[X]/(X^n+1) (inputs small)"""
    r = np.convolve(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
    out = np.array(r[:n], dtype=object)
    out[:n - 1] -= r[n:]
    return np.array([int(v) for v in out], dtype=object)

def qmul(a, b):
    return np.array([int(v) % q for v in zmul(np.asarray(a) % q, np.asarray(b) % q)], dtype=np.int64)

def matvec(A, v):
    out = np.zeros((A.shape[0], n), dtype=np.int64)
    for i in range(A.shape[0]):
        acc = np.zeros(n, dtype=np.int64)
        for j in range(A.shape[1]):
            acc = (acc + qmul(A[i, j], v[j])) % q
        out[i] = acc
    return out

def mono(a):
    v = np.zeros(n, dtype=object); a %= 2 * n
    if a < n: v[a] = 1
    else: v[a - n] = -1
    return v

def power2round(t):
    t = t % q; t0 = centered(t, 1 << d); return (t - t0) // (1 << d), t0

def decompose(rr):
    rr = np.asarray(rr) % q
    r0 = centered(rr, alpha)
    wrap = (rr - r0) == q - 1
    r1 = np.where(wrap, 0, (rr - r0) // alpha)
    r0 = np.where(wrap, r0 - 1, r0)
    return r1, r0

def highbits(rr): return decompose(rr)[0]

def usehint(h, rr):
    m = (q - 1) // alpha
    r1, r0 = decompose(rr)
    return np.where(h == 1, np.where(r0 > 0, (r1 + 1) % m, (r1 - 1) % m), r1)

def challenge(msg, w1):
    seed = hashlib.shake_256(msg + np.asarray(w1, dtype=np.int64).tobytes()).digest(16)
    st = np.random.default_rng(int.from_bytes(seed[:8], 'big'))
    c = np.zeros(n, dtype=object)
    for i, s in zip(st.choice(n, tau, replace=False), st.choice([-1, 1], tau)):
        c[i] = int(s)
    return c

def verify(pk, msg, sig):
    A, t1 = pk; c, z, h = sig
    if np.abs(centered(z)).max() >= gamma1 - beta: return False
    if h.sum() > omega: return False
    ct1 = np.array([qmul(c, t1[i]) for i in range(k)])
    rr = (matvec(A, z % q) - ct1 * (1 << d)) % q
    return all(int(x) == int(y) for x, y in zip(challenge(msg, usehint(h, rr)), c))

# ---------- Lagrange coefficients at monomial points, scaled by 2^(T-1) ----------
_zeta = 1753
_roots = [pow(_zeta, 2 * i + 1, q) for i in range(n)]
_P = np.array([[pow(r, j, q) for j in range(n)] for r in _roots], dtype=object)
_Pinv = np.array([[pow(pow(r, q - 2, q), m, q) for m in range(n)] for r in _roots], dtype=object)
_ninv = pow(n, q - 2, q)

def _fwd(a):
    a = [int(x) % q for x in a]
    return [sum(a[j] * _P[i, j] for j in range(n)) % q for i in range(n)]

def _inv(h):
    return np.array([(_ninv * sum(int(h[i]) * _Pinv[i, m] for i in range(n))) % q for m in range(n)])

def scaled_lagrange(exps):
    """mu_i = 2^(T-1) * lambda_i  as exact integer polys, for points X^{e_i}"""
    T = len(exps)
    ph = [_fwd(mono(e)) for e in exps]
    mus = []
    for i in range(T):
        num = [1] * n; den = [1] * n
        for j in range(T):
            if j == i: continue
            num = [num[s] * ph[j][s] % q for s in range(n)]
            den = [den[s] * (ph[j][s] - ph[i][s]) % q for s in range(n)]
        lam = [num[s] * pow(den[s], q - 2, q) % q for s in range(n)]
        mu = centered(_inv([x * pow(2, T - 1, q) % q for x in lam]))
        mus.append(np.array([int(x) for x in mu], dtype=object))
    return mus
