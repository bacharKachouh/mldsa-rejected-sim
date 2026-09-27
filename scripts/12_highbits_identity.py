"""
12 -- HighBits identity (paper, Section "Application to threshold ML-DSA"; Appendix item 12).
For r in Z_q let u = r + gamma2 - 1 and v = floor(u / 2^9).  Since alpha = 2^9 * 1023 and v < 2^15,
    floor(u / alpha) = floor((v + floor(v / 2^10) + 1) / 2^10),
and the FIPS high part is this value modulo 16.  Checked for every r in Z_q against Decompose.
Run:  python scripts/12_highbits_identity.py        (a few seconds)
"""
import numpy as np
from params import q, gamma2, alpha, decompose

r = np.arange(q, dtype=np.int64)
u = r + gamma2 - 1
v = u >> 9
ident = (v + (v >> 10) + 1) >> 10
assert np.array_equal(ident, u // alpha), "short form differs from floor(u / alpha)"
r1, _ = decompose(r)
ok = np.array_equal(ident % 16, r1)
print(f"floor(u/alpha) identity holds for all {q} residues; FIPS high part matches: {ok}")
print("PASS" if ok else "FAIL")
