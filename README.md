# Rejected ML-DSA-65 attempts: reproduction code

Code for every computed number in the paper *Statistical Simulation of Rejected Signing
Attempts in Unmodified ML-DSA-65* (B. Kachouh). Each script matches one item of the paper's
appendix "The computations". The scripts use plain Python and NumPy and share only
`scripts/params.py`, which holds the ML-DSA-65 parameters and Decompose as specified in FIPS 204.

```
pip install -r requirements.txt
python scripts/01_decompose_counts.py
```

Run every script from the repository root. Runtimes are for one desktop core. Slot sets are
printed in the paper's labelling, slot i having the root 1753^(2i+1) mod q.

| Script | Paper | What it checks | Time |
|---|---|---|---|
| `01_decompose_counts.py` | Lemma "bad counts are shift-independent" | 393 / 394 bad residues per cell for every shift | 1 min |
| `13_interval_structure.py` | Lemma "Fourier factor" | every good set is one cyclic interval, every bad set at most two, 6289 bad elements, for every shift | 3 min |
| `14_min_entropy.py` | Lemma "min-entropy of the commitment" | law of d(A) in exact rationals; Pr[d >= 20] = 2^-362.1 and 844.8 bits | seconds |
| `15_other_parameter_sets.py` | Section "What is not proved", Table "other parameter sets" | shift-independent bad counts, diagonal margin and per-slot exponent for ML-DSA-44, -65, -87 (indicative) | 3 min |
| `02_typical_pairs.py [N]` | Fact "typical pairs" | no rank-deficient slot in N random nonce pairs; flagged-coordinate frequency | < 1 min |
| `03a_toy_chi2.py` | toy-scale collision form | exact chi^2 by enumeration at toy parameters | minutes |
| `03b_toy_pair_weights.py [1] [2]` | atypical-pair weights | per-pair weights at toy scale; NTT zeros of short polynomials | minutes |
| `04a_coset_constants_and_block_small.py [A] [B3] [B4]` | Lemma "one dependent slot", Table "coset constants"; block-small searches | M_s for phi_64 = min(1, 64/\|a\|) and for the cells alone; block-small maxima with phi_64 | 1 min per s (A); 3 min (B3, B4) |
| `04b_coset_constants_independent.py` | Table "coset constants" | second implementation, different conventions (`PHI_K=64` default) | 1 min per s |
| `06_atypical_mass.py` | Theorems "unconditional atypical mass" and "unconditional bound" | the full (s, slot kinds) accounting with the refined Fourier factor and cut-off r <= 40, without certificates, with the s = 2 certificate and with both; the coset accounting; the final distance | a few minutes |
| `07_s2_certificate.py` | Lemma "s = 2 certificate" | certified bound over all 32 640 two-slot sets: exact integer minimum of the box terms, rational tails | 1 min |
| `08_s3_certificate.py a b` / `splits k` / `combine` | Lemma "s = 3 certificate" (used by the main theorem) | certified bound over all 2 763 520 three-slot sets: integer search, exact rational survivors, rational tails | about 18 CPU-hours in total |
| `16_mass_certified.py` | Theorems "unconditional atypical mass", "unconditional bound" and "existential unforgeability with revealed attempts" | the sum of `06` with directed rounding: interval arithmetic for every transcendental quantity, outward-rounded float sums, certified upper bounds on Lambda; checks every constant the theorems state | a few minutes |
| `17_sampleinball_uniform.py` | Lemma "SampleInBall" | exhaustive check at small (n, tau) that the FIPS 204 SampleInBall loop maps uniform choices to the uniform law on B_tau | seconds |

`rigorous.py` holds the rational upper bounds shared by the two certificates. No floating point
enters either certified bound: floats are used only to shortlist candidates, which are then
re-evaluated in integer or rational arithmetic. The certificates are computed with
phi_64 = min(1, 64/|a|), which is at least the refined factor of the paper at every point, so
they remain valid for it.

The s = 3 certificate is split across processes by first slot index. `python
scripts/08_s3_certificate.py splits 20` prints 20 chunks of roughly equal work; run
`08_s3_certificate.py a b` for each chunk from one working directory (each writes
`s3_chunk_a_b.json` holding its worst value as an exact fraction), then run `combine` in that
directory. It checks that the chunks cover every set and prints the certified bound.

`results/` holds the outputs of every script above, including the 20 per-chunk files of the
s = 3 run, so that `combine` can be re-checked without the 18-hour search, and
`s3_survivors.txt`, the list of all 96 terms of part (iii) of the s = 3 certificate (slot set and
block x). `python independent_checks/s3_check.py --witness` re-verifies every listed term, checks
the per-chunk counts against `s3_chunks/` and recomputes the largest value, in seconds.

## Independent checks

`independent_checks/` holds second implementations, written separately and importing nothing
from `scripts/`, of five items: `r0_counts.py` (Lemma "bad counts are shift-independent"),
`Ms.py` (the coset constants, through a product over the whole subgroup), `s2_check.py` (the
s = 2 certificate, by an integer threshold test), `s3_check.py` (the s = 3 certificate: the
block-step matrix formed by diagonalising in the slot basis, and the meet-in-the-middle search
run from the other side, on the last row; about 3 CPU-hours, 14 minutes on 23 cores) and
`mass_check.py` (the atypical mass, with Lambda summed piecewise and Hurwitz-zeta tails).
`expected_output.txt` records their output.

## Other material

`scripts/09a` to `12`, `core.py`, `mldsa_ref.py`, `threshold_openw1.py` and `bench/` concern a
threshold protocol and its MPC benchmarks. The current version of the paper does not use them.

## License

MIT; see `LICENSE`.
