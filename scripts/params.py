"""ML-DSA-65 parameters (FIPS 204, Table 1) and Decompose exactly as specified (Algorithm 36)."""
import numpy as np

n, q = 256, 8380417
k, l, eta = 6, 5, 4
gamma1 = 1 << 19
gamma2 = (q - 1) // 32
alpha = 2 * gamma2
tau, beta, d, omega = 49, 196, 13, 55
zeta = 1753                                   # primitive 512-th root of unity mod q


def centered(x, m=q):
    r = np.asarray(x) % m
    return np.where(r > m // 2, r - m, r)


def decompose(rr):
    """(r1, r0) with r = r1*alpha + r0, including the FIPS exceptional case r - r0 = q - 1."""
    rr = np.asarray(rr) % q
    r0 = centered(rr, alpha)
    wrap = (rr - r0) == q - 1
    r1 = np.where(wrap, 0, (rr - r0) // alpha)
    r0 = np.where(wrap, r0 - 1, r0)
    return r1, r0


def slot_matrix():
    """P[i, j] = zeta^((2i+1) j) mod q, so (P @ y) % q evaluates y at the 256 slot roots."""
    roots = [pow(zeta, 2 * i + 1, q) for i in range(n)]
    return np.array([[pow(r, j, q) for j in range(n)] for r in roots], dtype=np.int64)
