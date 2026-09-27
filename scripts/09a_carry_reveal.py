"""
09a -- carry-reveal reconstruction (paper: Lemma "the carry must stay private", Appendix item 9).
If the carry m (or Q) is revealed together with the public w1, a coalition reconstructs
LowBits(w) - sum_{honest} LowBits(w_i) exactly.  10 random instances each for
(N, |H|) = (2, 1), (3, 2), (8, 4).  The key-recovery count uses the law of 09b.
Run:  python scripts/09a_carry_reveal.py
"""
import math
import numpy as np
from core import (n, q, k, l, eta, gamma1, gamma2, alpha, tau, beta, d, omega,
                  centered, zmul, qmul, matvec, decompose, highbits)
from mldsa_ref import keygen

def hdr(t): print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)
def line(t): print("  " + t)

def check_carry_output_leak():
    """Revealing the carry m (or Q, or H) together with the public
    w1 lets a T-1 coalition compute  LowBits(w) - sum_{honest} LowBits(w_i)  EXACTLY.
    That is a noisy disclosure of LowBits(w) with noise std alpha*sqrt(|H|/12); the
    law of 09b then gives key recovery in ~2^32 signatures."""
    hdr("Carry-output leak: m + w1 => exact  LowBits(w) - L_H")
    rng = np.random.default_rng(6); pk, sk = keygen(rng); A = sk[0]
    def cd(w): w = w % q; return w // alpha, w % alpha
    allok = True
    for T, nH in ((2, 1), (3, 2), (8, 4)):
        good = 0; trials = 10
        for _ in range(trials):
            ys = [rng.integers(0, q, size=(l, n), dtype=np.int64) for _ in range(T)]
            wis = [matvec(A, y) for y in ys]
            hon, cor = wis[:nH], wis[nH:]
            H = sum(cd(w)[0].astype(object) for w in wis); L = sum(cd(w)[1].astype(object) for w in wis)
            WZ = sum(w.astype(object) for w in wis); Q = WZ // q; m = (L - Q) // alpha
            w = WZ % q; w1 = w // alpha; ell = w % alpha                      # w1 public, ell secret
            # --- adversary view: m, Q (F_carry output), w1, and its own corrupt w_j ---
            V = (w1 + 16 * Q - m) - sum(cd(w)[0].astype(object) for w in cor)  # = sum_{honest} HighBits(w_i)
            u = sum(w.astype(object) for w in cor) % q                          # sum of corrupt w_j mod q
            base = alpha * (V - w1) + u                                           # = ell - L_H + q*(kappa+Q_H)
            # ell - L_H lies in (-nH*alpha, alpha): unique representative
            lo_bound = -nH * alpha
            guess = base - q * ((base - lo_bound) // q)
            truth = ell.astype(object) - sum(cd(w)[1].astype(object) for w in hon)
            good += np.array_equal(guess, truth)
        line(f"T={T:2d}, |H|={nH}: adversary recovers LowBits(w) - L_H exactly in {good}/{trials} trials")
        allok &= (good == trials)
    sigma = alpha * math.sqrt(2 / 12); S = (2 * sigma * 0.153) ** 2
    line(f"noise std at |H|=2 = 2^{math.log2(sigma):.1f}  => LS key recovery after ~2^{math.log2(S):.1f} signatures (law of 09b)")
    print(f"  => {'PASS' if allok else 'FAIL'} (leak confirmed: F_carry must output w1 only)")
    return allok

if __name__ == "__main__":
    check_carry_output_leak()
