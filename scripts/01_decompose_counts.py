"""
01 -- FIPS-exact rejection counts (paper: Lemma "bad counts are shift-independent", Appendix item 1).

Every residue r in [0, q) goes through Decompose as written in FIPS 204.  Cell C_v = {r : r1 = v}.
A residue r' is bad if |r0(r')| >= gamma2 - beta.  For every cell v and every shift |s| <= beta
we count N_v(s) = #{r in C_v : (r - s) mod q is bad}.  Expected: 393 for v != 0, 394 for v = 0.
Run:  python scripts/01_decompose_counts.py          (about a minute)
"""
import time
import numpy as np
from params import q, gamma2, beta, decompose

t0 = time.time()
r = np.arange(q, dtype=np.int64)
r1, r0 = decompose(r)
bad = (np.abs(r0) >= gamma2 - beta).astype(np.int64)
sizes = np.bincount(r1, minlength=16)
print(f"cell sizes: v=0 -> {sizes[0]}, v=1..15 -> {sorted(set(sizes[1:].tolist()))}")
seen = set()
for s in range(-beta, beta + 1):
    shifted_bad = bad[(r - s) % q]
    counts = np.bincount(r1, weights=shifted_bad, minlength=16).astype(np.int64)
    seen.add((int(counts[0]), tuple(sorted(set(counts[1:].tolist())))))
print(f"distinct (N_0, {{N_1..N_15}}) over all {2 * beta + 1} shifts: {sorted(seen)}")
ok = seen == {(2 * beta + 2, (2 * beta + 1,))}
print(f"{'PASS' if ok else 'FAIL'}: N_v(s) = {2 * beta + 1} for v != 0 and {2 * beta + 2} for v = 0, every shift  [{time.time() - t0:.0f}s]")
