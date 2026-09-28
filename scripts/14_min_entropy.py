"""
14 -- min-entropy of the commitment (Lemma "min-entropy of the commitment").

For the upper 5x5 block A_1 of A let d(A) = sum over the 256 slots of the corank of A_1(zeta_i).
Then  max_w Pr_y[HighBits(Ay) = w] <= q^d(A) ((alpha+1)/(2 gamma_1))^1280.
For uniform A the A_1(zeta_i) are independent uniform 5x5 matrices over Z_q.  This script computes
the law of the corank of such a matrix from the standard count of matrices of each rank, takes the
256-fold convolution in exact rational arithmetic (truncated at d = 60), and prints Pr[d >= D]
together with the resulting min-entropy bound.  Output: Pr[d >= 20] = 2^-362.1 and 844.8 bits.
Run:  python scripts/14_min_entropy.py        (a few seconds)
"""
from fractions import Fraction as Fr
import math
q = 8380417; n = 5; alpha = 523776; gamma1 = 2**19

def n_rank(r):
    """number of n x n matrices of rank r over F_q."""
    num = 1
    for i in range(r): num *= (q**n - q**i) ** 2
    den = 1
    for i in range(r): den *= (q**r - q**i)
    return num // den

pc = [Fr(n_rank(n - c), q**(n * n)) for c in range(n + 1)]          # law of the corank
assert sum(pc) == 1
print(f"log2 Pr[corank >= 1] = {math.log2(float(1 - pc[0])):.3f},  log2 Pr[corank >= 2] = {math.log2(float(sum(pc[2:]))):.3f}")
D = 60
dist = [Fr(1)] + [Fr(0)] * D
for _ in range(256):
    new = [Fr(0)] * (D + 1)
    for a, pa in enumerate(dist):
        if pa:
            for c in range(n + 1):
                if a + c <= D: new[a + c] += pa * pc[c]
    dist = new
base = 1280 * math.log2(2 * gamma1 / (alpha + 1))
print(f"min-entropy with d = 0: {base:.2f} bits")
for Dc in (5, 10, 20, 30, 40):
    tail = 1 - sum(dist[:Dc])
    print(f"Pr[d >= {Dc}] = 2^{math.log2(float(tail)):.1f};  for d <= {Dc - 1}: H_inf >= {base - (Dc - 1) * math.log2(q):.1f} bits")
