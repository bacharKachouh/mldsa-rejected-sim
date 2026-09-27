"""Tests for the host-side ML-DSA pieces used by the MPC benchmark."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mldsa_host as mh
from core import q, alpha, gamma2, k, l, n, beta, decompose, highbits, zmul, tau
from mldsa_ref import makehint


def boundary_sweep():
    pts = []
    for t in range(17):
        c = t * alpha - gamma2
        pts += list(range(c - 2, c + 3))
    pts += list(range(q - gamma2 - 3, q))
    pts += [0, 1, gamma2 - 1, gamma2, gamma2 + 1]
    return np.array(sorted({p for p in pts if 0 <= p < q}), dtype=np.int64)


def test_w1_thresholds_match_fips_on_boundaries():
    r = boundary_sweep()
    w1, r0 = mh.w1_via_thresholds(r)
    r1_ref, r0_ref = decompose(r)
    assert np.array_equal(w1, r1_ref)
    assert np.array_equal(r0, r0_ref)


def test_w1_thresholds_match_fips_random():
    r = np.random.default_rng(1).integers(0, q, size=200_000, dtype=np.int64)
    w1, r0 = mh.w1_via_thresholds(r)
    r1_ref, r0_ref = decompose(r)
    assert np.array_equal(w1, r1_ref) and np.array_equal(r0, r0_ref)


def _random_c(rng):
    c = np.zeros(n, dtype=np.int64)
    idx = rng.choice(n, tau, replace=False)
    c[idx] = rng.choice([-1, 1], tau)
    return c


def test_hint_thresholds_match_makehint():
    """h = [r0' > hi] + [r0' <= lo] equals FIPS MakeHint(-ct0, w - cs2 + ct0) whenever the r0-test passes."""
    rng = np.random.default_rng(2)
    checked = 0
    for trial in range(400):
        w = rng.integers(0, q, size=n, dtype=np.int64)
        if trial % 4 == 0:        # force cell 0, including the top cell
            w = rng.choice(np.r_[np.arange(q - gamma2, q), np.arange(0, gamma2 + 1)], n)
        cs2 = rng.integers(-beta, beta + 1, size=n, dtype=np.int64)
        ct0 = rng.integers(-gamma2 + 1, gamma2, size=n, dtype=np.int64)
        w1, r0 = decompose(w)
        r0p = r0 - cs2
        ok = np.abs(r0p) < gamma2 - beta
        hi, lo = mh.hint_thresholds(w1, ct0)
        h_mpc = (r0p > hi).astype(np.int64) + (r0p <= lo).astype(np.int64)
        h_ref = makehint((-ct0) % q, (w - cs2 + ct0) % q)
        assert np.array_equal(h_mpc[ok], h_ref[ok])
        checked += ok.sum()
    assert checked > 50_000


def test_hint_threshold_exact_edge_w1_zero():
    """w1 = 0 and offset exactly -gamma2: FIPS keeps HighBits 0 (top cell), so no hint."""
    w1 = np.array([0, 3])
    ct0 = np.array([0, 0])
    hi, lo = mh.hint_thresholds(w1, ct0)
    r0p = np.array([-gamma2, -gamma2])
    h = (r0p > hi).astype(int) + (r0p <= lo).astype(int)
    assert list(h) == [0, 1]


def test_keygen_and_honest_signature_verifies():
    key = mh.keygen(seed=5)
    rng = np.random.default_rng(6)
    msg = b"bench"
    for _ in range(200):
        y = rng.integers(-(1 << 19) + 1, (1 << 19) + 1, size=(l, n), dtype=np.int64)
        w = mh.matvec_int(key['A'], y) % q
        w1 = highbits(w)
        c = mh.challenge_for(msg, w1)
        jstar, _ = mh.fips_first_accepting(key, [y], [w], [c])
        if jstar == 0:
            z = y + mh.polymul_vec(c, key['s1'])
            sig, ok = mh.assemble_and_verify(key, msg, c, z)
            assert ok
            return
    raise AssertionError("no accepting attempt in 200 tries")
