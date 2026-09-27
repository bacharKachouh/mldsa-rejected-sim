# Rejected ML-DSA-65 attempts: reproduction code

Code for every computed number in the paper *Rejected Signing Attempts in Unmodified ML-DSA*
(B. Kachouh). Each script matches one item of the paper's appendix "The computations". The
scripts use plain Python and NumPy and share only `scripts/params.py`, which holds the
ML-DSA-65 parameters and Decompose as specified in FIPS 204.

```
pip install -r requirements.txt
python scripts/01_decompose_counts.py
```

Run every script from the repository root. Runtimes are for one desktop core.

| Script | Paper | What it checks | Time |
|---|---|---|---|
| `01_decompose_counts.py` | Lemma "bad counts are shift-independent" | 393 / 394 bad residues per cell for every shift | 1 min |
| `02_typical_pairs.py [N]` | Fact "typical pairs" | no rank-deficient slot in N random nonce pairs; flagged-coordinate frequency | < 1 min |
| `03a_toy_chi2.py` | toy-scale collision form | exact chi^2 by enumeration at toy parameters | minutes |
| `03b_toy_pair_weights.py [1] [2] [3]` | atypical-pair weights | per-pair weights at toy scale; NTT zeros of short polynomials; character bounds | minutes |
| `04a_coset_constants_and_block_small.py [A] [B3] [B4]` | Table "coset constants"; block-small searches | M_s for phi = min(1, 64/\|a\|) and min(1, 8/\|a\|); block-small maxima | 1 min per s (A); longer for B |
| `04b_coset_constants_independent.py` | Table "coset constants" | second implementation, different conventions (`PHI_K=64` default) | 1 min per s |
| `06_atypical_mass.py` | Theorems "unconditional atypical mass" and "unconditional bound" | the full (s, slot kinds) accounting with and without certificates, the coset accounting, and the final distance | 12 min |
| `07_s2_certificate.py` | Lemma "s = 2 certificate" | certified bound over all 32 640 two-slot sets: exact integer minimum of the box terms, rational tails | 1 min |
| `08_s3_certificate.py a b` / `splits k` / `combine` | Lemma "s = 3 certificate" | certified bound over all 2 763 520 three-slot sets: integer search, exact rational survivors, rational tails | about 18 CPU-hours in total |

`rigorous.py` holds the rational upper bounds shared by the two certificates. No floating point
enters either certified bound: floats are used only to shortlist candidates, which are then
re-evaluated in integer or rational arithmetic.

The s = 3 certificate is split across processes by first slot index. `python
scripts/08_s3_certificate.py splits 20` prints 20 chunks of roughly equal work; run
`08_s3_certificate.py a b` for each chunk from one working directory (each writes
`s3_chunk_a_b.json` holding its worst value as an exact fraction), then run `combine` in that
directory. It checks that the chunks cover every set and prints the certified bound.

`results/` holds the outputs of the two certificates as run for the paper, including the 20
per-chunk files of the s = 3 run, so that `combine` can be re-checked without the 18-hour search.
