"""
10 -- nonce flatness (paper: Lemma "the nonce must be box-uniform", Appendix item 10).
Exact E[z | accept] at v = (c s1)_j = beta for a box-uniform nonce (0) and for the sum of two
half-width boxes (about v), and the least-squares signature count for the second case.
Run:  python scripts/10_nonce_flatness.py
"""
import math
import numpy as np
from core import (n, q, k, l, eta, gamma1, gamma2, alpha, tau, beta, d, omega,
                  centered, zmul, qmul, matvec, decompose, highbits)
from mldsa_ref import keygen

def hdr(t): print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)
def line(t): print("  " + t)

def check_nonce_flatness():
    """Rejection sampling is key-independent ONLY for a box-uniform
    nonce.  Exact 1-D computation of E[z | accept] for v = (c*s1)_j = beta:  uniform y gives
    exactly 0; the sum-of-two-boxes nonce gives ~v, i.e. z averages to c*s1."""
    hdr("Nonce flatness: E[z | accept] must not depend on c*s1")
    v = beta; B = gamma1 - beta
    ys = np.arange(-gamma1 + 1, gamma1 + 1, dtype=np.int64)          # FIPS box
    z = ys + v; acc = np.abs(z) < B
    mean_uniform = z[acc].mean()
    # sum of two boxes [-gamma1/2, gamma1/2): triangular density on the same support
    half = gamma1 // 2
    ys2 = np.arange(-gamma1, gamma1 + 1, dtype=np.int64)
    dens = np.clip(gamma1 - np.abs(ys2), 0, None).astype(np.float64)   # triangular weights
    z2 = ys2 + v; acc2 = np.abs(z2) < B
    mean_tri = float((z2[acc2] * dens[acc2]).sum() / dens[acc2].sum())
    line(f"v = (c s1)_j = {v}")
    line(f"box-uniform nonce (FIPS):    E[z | accept] = {mean_uniform:+.3f}   (exactly 0: key-independent)")
    line(f"sum-of-2-boxes nonce:         E[z | accept] = {mean_tri:+.3f}   (~v: z leaks c*s1)")
    Var = gamma1 ** 2 / 6; S = 16 * Var / tau
    line(f"=> LS recovery of s1 from ~2^{math.log2(S):.1f} signatures at T=2 (Var(y)=gamma1^2/(3T))")
    ok = abs(mean_uniform) < 1e-6 and abs(mean_tri - v) < 1.0
    print(f"  => {'PASS' if ok else 'FAIL'} (only a box-uniform aggregate nonce is admissible)")
    return ok

if __name__ == "__main__":
    check_nonce_flatness()
