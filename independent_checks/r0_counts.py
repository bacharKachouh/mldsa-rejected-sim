"""
r0_counts.py -- independent check of Lemma "bad counts are shift-independent" (Lemma r0).
Decompose is written from FIPS 204 Alg. 36 (no repo imports). For every cell v of HighBits and every
shift |s| <= beta = 196, counts N_v(s) = #{r in C_v : |LowBits((r - s) mod q)| >= gamma2 - beta}.
Expected: N_0 = 394 and N_1..N_15 = 393 for all 393 shifts; the claim fails at s = -197.
Run: python r0_counts.py      (about 25 s)
"""
import numpy as np, time
q = 8380417; g2 = (q - 1) // 32; al = 2 * g2; beta = 196
t0 = time.time()
r = np.arange(q, dtype=np.int64)
r0 = r % al; r0 = np.where(r0 > al // 2, r0 - al, r0)
ex = (r - r0) == q - 1
r1 = np.where(ex, 0, (r - r0) // al); r0 = np.where(ex, r0 - 1, r0)
bad = (np.abs(r0) >= g2 - beta).astype(np.int32)
def counts(s):
    c = np.bincount(r1, weights=np.roll(bad, s), minlength=16).astype(int)   # roll: bad[(r - s) mod q]
    return int(c[0]), tuple(sorted(set(c[1:].tolist())))
seen = {counts(s) for s in range(-beta, beta + 1)}
print("cell sizes |C_0|, |C_v|:", int((r1 == 0).sum()), int((r1 == 1).sum()))
print("distinct (N_0, {N_1..N_15}) over |s| <= 196:", sorted(seen), "PASS" if seen == {(394, (393,))} else "FAIL")
for s in (197, -197):
    print(f"outside the range, s = {s}:", counts(s))
print(f"[{time.time()-t0:.0f}s]")
