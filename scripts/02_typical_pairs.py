"""
02 -- Typical pairs at full scale (paper: Fact "typical pairs", Appendix item 2).

(a) For N independent nonce pairs (y, y') uniform on the box [-gamma1+1, gamma1]^{l x 256}, compute
    the rank over Z_q of the l x 2 slot matrix [y(zeta_i) | y'(zeta_i)] for all 256 slots and
    report how many slots have rank below two (expected: none; the probability is < 256 * 2^-75).
(b) For the same nonces and N uniform matrices A (NTT domain), report the frequency of bad
    coordinates of w = Ay (the flag of the r0-data with s2 = 0) against the exact value
    ((2 beta + 1) * 15 * alpha + (2 beta + 2) * (alpha + 1)) / (q * alpha-cells) from Corollary D'.
Run:  python scripts/02_typical_pairs.py [N]        (default N = 60, under a minute)
"""
import sys, time
import numpy as np
from params import q, n, k, l, gamma1, gamma2, beta, alpha, decompose, slot_matrix

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rng = np.random.default_rng(2026)
P = slot_matrix()
Pinv = None

def evals(y):                       # y: (l, n) small ints -> (l, n) slot values mod q
    out = np.zeros_like(y)
    for m in range(y.shape[0]):     # entries < 2^19 * 2^23 * 256 < 2^63
        out[m] = (P @ (y[m] % q)) % q
    return out

def inverse_slots():
    """Inverse of the evaluation map: coefficients = P^-1 values (exact, via Python ints once)."""
    roots = [pow(1753, 2 * i + 1, q) for i in range(n)]
    ninv = pow(n, q - 2, q)
    # (P^-1)[j, i] = n^-1 * r_i^-j
    return np.array([[ninv * pow(pow(r, q - 2, q), j, q) % q for r in roots] for j in range(n)], dtype=np.int64)

t0 = time.time()
Pinv = inverse_slots()
low_rank = 0; bad = 0; total = 0
for trial in range(N):
    y = rng.integers(-gamma1 + 1, gamma1 + 1, size=(l, n), dtype=np.int64)
    y2 = rng.integers(-gamma1 + 1, gamma1 + 1, size=(l, n), dtype=np.int64)
    Y, Y2 = evals(y), evals(y2)
    minors = np.zeros(n, dtype=bool)
    for a in range(l):
        for b in range(a + 1, l):
            minors |= ((Y[a] * Y2[b] - Y[b] * Y2[a]) % q) != 0      # products < 2^46
    low_rank += int((~minors).sum())
    A = rng.integers(0, q, size=(k, l, n), dtype=np.int64)         # A in the NTT domain, uniform
    for row in range(k):
        wv = np.zeros(n, dtype=np.int64)
        for c in range(l):
            wv = (wv + A[row, c] * Y[c]) % q
        w = np.zeros(n, dtype=np.int64)
        for j0 in range(0, n, 32):                                   # chunked to stay below 2^63
            w = (w + Pinv[:, j0:j0 + 32] @ wv[j0:j0 + 32]) % q
        _, r0 = decompose(w)
        bad += int((np.abs(r0) >= gamma2 - beta).sum()); total += n
exact = (15 * alpha * (2 * beta + 1) / alpha + (alpha + 1) * (2 * beta + 2) / (alpha + 1)) / q  # sum_v |C_v| p_v / q
print(f"(a) {N} nonce pairs, {N * n} slots: slots of rank < 2 = {low_rank}")
print(f"(b) flagged-coordinate frequency {bad / total:.3e} over {total} coordinates;  exact {exact:.3e}  [{time.time() - t0:.0f}s]")
