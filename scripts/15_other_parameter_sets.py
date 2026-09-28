"""
15 -- indicative margins for ML-DSA-44 and ML-DSA-87 (Section "What is not proved", Table "other
parameter sets").  Not used by any theorem; the full accounting is done only for ML-DSA-65.

For each parameter set this script
  (a) checks the analogue of Lemma "bad counts are shift-independent": for every shift |s| <= beta
      and every cell, the number of residues whose shifted low part fails the r0-test, with
      Decompose written from FIPS 204 for that gamma_2;
  (b) prints the diagonal margin  l*256*log2(2 gamma_1) - 256 k log2(q / min bad count),
      i.e. nonce min-entropy minus log2 max 1/D' (Lemma "diagonal" with r = 0);
  (c) forms the refined Fourier factor of Lemma "Fourier factor" for that cell structure,
      phi(a) = min(1, (m/2)/|a| + min(bad/q, m/|a|)) with m cells and bad = total bad count,
      and prints Lambda(L) = sum_a phi(a)^L for large L;
  (d) prints the per-slot exponents: probability log2((q+1) (2 gamma_1)^-l) of one dependent
      slot (Lemma "probability of dependent slots", uniform case) and weight k log2 Lambda(128),
      and their sum.
Run:  python scripts/15_other_parameter_sets.py        (about three minutes)
"""
import math
import numpy as np

q = 8380417
SETS = {                      # k, l, eta, tau, gamma1, gamma2
    "ML-DSA-44": (4, 4, 2, 39, 2**17, (q - 1) // 88),
    "ML-DSA-65": (6, 5, 4, 49, 2**19, (q - 1) // 32),
    "ML-DSA-87": (8, 7, 2, 60, 2**19, (q - 1) // 32),
}
r = np.arange(q, dtype=np.int64)

for name, (k, l, eta, tau, g1, g2) in SETS.items():
    beta = tau * eta; alpha = 2 * g2
    u = r % alpha
    r0 = np.where(u <= alpha // 2, u, u - alpha)
    exc = (r - r0) == q - 1
    hb = np.where(exc, 0, (r - r0) // alpha)
    lb = np.where(exc, r0 - 1, r0)
    bad = np.abs(lb) >= g2 - beta
    m = int(hb.max()) + 1
    counts = set(); totals = set()
    cell_idx = [hb == v for v in range(m)]
    for s in range(-beta, beta + 1):
        b = bad[(r - s) % q]
        cs = tuple(int(np.count_nonzero(c & b)) for c in cell_idx)
        counts.add(tuple(sorted(set(cs)))); totals.add(sum(cs))
    minbad = min(min(c) for c in counts); total = max(totals)
    shift_indep = len(counts) == 1 and len(totals) == 1
    diag = l * 256 * math.log2(2 * g1) - 256 * k * math.log2(q / minbad)
    a = np.arange(1, q // 2 + 1, dtype=np.float64)
    lphi = np.log(np.minimum(1.0, (m / 2) / a + np.minimum(total / q, m / a)))
    lam = math.log2(1.0 + 2.0 * float(np.exp(128 * lphi).sum()))
    prob = math.log2(q + 1) - l * math.log2(2 * g1)
    print(f"{name}: beta={beta}, {m} cells, bad counts per cell {sorted(counts)} "
          f"(shift-independent: {shift_indep}), total bad {total}")
    print(f"   diagonal margin {diag:.0f} bits;  Lambda(128) = 2^{lam:.2f};  per slot: probability 2^{prob:.1f},"
          f" weight 2^{k * lam:.1f}, together 2^{prob + k * lam:.1f}")
