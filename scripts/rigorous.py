"""
rigorous.py -- certified (rational) upper bounds used by the s = 2 and s = 3 certificates.

No floating point enters any bound returned here.  Every quantity is a fractions.Fraction that is
an upper bound on the real number it stands for, and log2_up returns a decimal k such that the
Fraction is at most 2^k, verified by exact integer comparison.

  phi(a) = min(1, 64/|a|_c)  on Z_q, q = 8380417.

  S(p, a0)      >= sum_{a >= a0} (64/a)^p          (a0 >= 65; the sum over a0..(q-1)/2 is smaller)
  Lambda_up(p)  >= Lambda(p) = sum_{a in Z_q} phi(a)^p = 129 + 2 sum_{a=65}^{(q-1)/2} (64/a)^p
  tail_up(p)    >= sum_{|a|_c > 128} phi(a)^p = 2 sum_{a=129}^{(q-1)/2} (64/a)^p

S(p, a0) sums the terms a0..K exactly, each rounded UP to a multiple of 2^-E, and bounds the rest
by the integral int_K^inf (64/x)^p dx = 64^p K^(1-p) / (p-1), valid because (64/x)^p decreases.
"""
from fractions import Fraction
import math

E_BITS = 4096          # fixed-point precision of the rounded-up terms
K_EXACT = 4000         # terms summed exactly before the integral remainder


def _ceil_div(a, b):
    return -((-a) // b)


def S(p, a0, K=K_EXACT):
    num = 64 ** p << E_BITS
    acc = 0
    for a in range(a0, K + 1):
        acc += _ceil_div(num, a ** p)
    exact_part = Fraction(acc, 1 << E_BITS)
    remainder = Fraction(64 ** p, (p - 1) * K ** (p - 1))
    return exact_part + remainder


def Lambda_up(p):
    return 129 + 2 * S(p, 65)


def tail_up(p):
    return 2 * S(p, 129)


def log2_up(F, step=Fraction(1, 100)):
    """Smallest k on the grid `step` with F <= 2^k, checked exactly: F^d <= 2^(k d) for k = n/d."""
    F = Fraction(F)
    assert F > 0
    d = step.denominator
    guess = math.floor((F.numerator.bit_length() - F.denominator.bit_length()) * d) - 2 * d
    lhs_num, lhs_den = F.numerator ** d, F.denominator ** d
    n = guess
    while True:                                   # increase until 2^(n/d) >= F
        if n >= 0:
            if lhs_num <= lhs_den << n:
                return Fraction(n, d)
        else:
            if lhs_num << (-n) <= lhs_den:
                return Fraction(n, d)
        n += 1


def fmt(k):
    return f"2^{float(k):.3f}"
