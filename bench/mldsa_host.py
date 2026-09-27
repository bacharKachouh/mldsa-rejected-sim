"""
mldsa_host.py -- the public (host-side) parts of the MPC benchmark pipeline, and a FIPS-exact
reference for what the MPC programs compute.

The MPC programs (bench/mpc/*.mpc) evaluate the ideal functionalities of the open-w1 construction
(paper, Section 9). Everything here is either public in that protocol (hash, hint, verification,
the per-coefficient hint thresholds derived from public w1 and c*t0) or a plaintext reference used
only to check the MPC output.
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "scripts"))

import numpy as np
from core import (n, q, k, l, eta, gamma1, gamma2, alpha, beta, d, omega,
                  power2round, decompose, challenge, verify, qmul)
from mldsa_ref import makehint


def keygen(seed):
    """Same distribution as scripts/threshold_openw1.Ideal.dkg (A uniform, s1, s2 eta-bounded)."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(k, l, n), dtype=np.int64)
    s1 = rng.integers(-eta, eta + 1, size=(l, n), dtype=np.int64)
    s2 = rng.integers(-eta, eta + 1, size=(k, n), dtype=np.int64)
    t = (matvec_int(A, s1) + s2) % q
    t1, t0 = power2round(t)
    return dict(A=A, s1=s1, s2=s2, t0=np.asarray(t0, dtype=np.int64), t1=np.asarray(t1, dtype=np.int64))


def polymul(a, b):
    """Exact negacyclic product over Z of two int64 polynomials (|result| < 2^63 assumed)."""
    r = np.convolve(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    out = r[:n].copy()
    out[:n - 1] -= r[n:]
    return out


def polymul_vec(c, s):
    """c * s row by row, c a single polynomial, s shape (rows, n)."""
    return np.array([polymul(c, row) for row in s], dtype=np.int64)


def matvec_int(A, y):
    """A*y over Z (not reduced). A entries in [0, q), |y| <= 2^19: |result| < 2^53."""
    return np.array([sum(polymul(A[i, j], y[j]) for j in range(A.shape[1])) for i in range(A.shape[0])],
                    dtype=np.int64)


def w1_via_thresholds(r):
    """The formula the MPC uses for FIPS Decompose on r in [0, q):
    b_t = [r > t*alpha - gamma2], t = 1..16;  w1 = sum_{t<=15} b_t - 15*b_16;
    r0 = r - alpha * sum_{t<=16} b_t - b_16  (b_16 is the FIPS top-cell case)."""
    r = np.asarray(r, dtype=np.int64)
    b = np.stack([(r > t * alpha - gamma2).astype(np.int64) for t in range(1, 17)])
    w1 = b[:15].sum(axis=0) - 15 * b[15]
    r0 = r - alpha * b.sum(axis=0) - b[15]
    return w1, r0


def hint_thresholds(w1, ct0):
    """Public per-coefficient thresholds for the hint test, valid once the r0-test has passed.
    With r0' = LowBits(w) - c*s2, the FIPS hint MakeHint(-ct0, w - cs2 + ct0) is
    h = [r0' > hi] + [r0' <= lo], hi = gamma2 - ct0, lo = -gamma2 - [w1 == 0] - ct0."""
    w1 = np.asarray(w1, dtype=np.int64)
    ct0 = np.asarray(ct0, dtype=np.int64)
    hi = gamma2 - ct0
    lo = np.where(w1 == 0, -gamma2 - 1, -gamma2) - ct0
    return hi, lo


def challenge_for(msg, w1):
    return np.array([int(v) for v in challenge(msg, w1)], dtype=np.int64)


def fips_first_accepting(key, ys, ws, cs):
    """FIPS-exact acceptance for a batch; returns (first accepting index or None, statuses).
    ys: centered nonces (l, n); ws: w = A*y mod q (k, n); cs: challenges."""
    statuses = []
    for j, (y, w, c) in enumerate(zip(ys, ws, cs)):
        z = y + polymul_vec(c, key['s1'])
        if np.abs(z).max() >= gamma1 - beta:
            statuses.append('reject-z'); continue
        cs2 = polymul_vec(c, key['s2'])
        if np.abs(decompose(w)[1] - cs2).max() >= gamma2 - beta:
            statuses.append('reject-r0'); continue
        ct0 = polymul_vec(c, key['t0'])
        h = makehint((-ct0) % q, (w - cs2 + ct0) % q)
        if h.sum() > omega:
            statuses.append('reject-hint'); continue
        statuses.append('accept')
        return j, statuses
    return None, statuses


def assemble_and_verify(key, msg, c, z):
    """Public step after the online MPC: hint from public data, signature, stock verification."""
    A, t1, t0 = key['A'], key['t1'], key['t0']
    z = np.asarray(z, dtype=np.int64)
    ct1 = np.array([qmul(c, t1[i]) for i in range(k)])
    rr = (matvec_int(A, z) - ct1 * (1 << d)) % q
    ct0 = polymul_vec(c, t0)
    h = makehint((-ct0) % q, rr)
    sig = (np.array(c, dtype=object), z % q, h)
    return sig, bool(verify((A, t1), msg, sig))
