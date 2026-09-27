"""
cost_model.py -- per-operation cost assumptions for the secure computation of one signing attempt (estimates).

Every number is derived from the explicit per-operation assumptions below; change them and
re-run.  All are [EXTRAPOLATED] order-of-magnitude figures, NOT measurements.

Model: dishonest-majority MPC with abort, GMW/TinyOT-style binary circuits for comparisons,
bit-decompositions and Keccak, with SPDZ-style arithmetic sharing over Z_q for the linear parts
(which are free) and edaBit/daBit conversions.  Online communication per AND per party is
2 bits to each other party (masked-value opening); offline cost is the authenticated AND
triple.  A BMR (constant-round) row is given for comparison.
"""
import math, sys

# ---------------- ML-DSA-65 ----------------
n, k, l = 256, 6, 5
NK, NL = n * k, n * l                       # 1536, 1280 coefficients
LOGQ = 23
ATTEMPTS = 5.0                              # reproduce.py check 1: mean 5.03
P_ZPASS = ((2 * (2**19 - 196) - 1) / (2 * 2**19 - 1)) ** NL   # ~0.62: r0-test only runs then
HASH_BYTES = 64 + NK * 4 // 8               # mu (64 B) + w1Encode (768 B) = 832 B
KECCAK_RATE = 136                           # SHAKE256
KECCAK_PERMS = math.ceil((HASH_BYTES + 1) / KECCAK_RATE)   # 7 (48-B c~ fits in 1st squeeze)
KECCAK_AND_PER_PERM = 24 * 1600             # chi: 1600 ANDs per round, 24 rounds
KECCAK_DEPTH_PER_PERM = 24

# ---------------- per-op cost assumptions [EXTRAPOLATED] ----------------
# (ANDs online-equivalent, AND-depth rounds).  Comparison of a 23-bit shared value via
# edaBit + binary adder: ~2k ANDs online for k=23 (adder + sign extraction).
CMP_AND, CMP_DEPTH = 50, 6
BITDEC_AND, BITDEC_DEPTH = 60, 6            # edaBit + one 23-bit adder
HIGHBITS_CMP = 4                            # 16-way threshold search on bits (alpha not power of 2)
B2A_DABITS_PER_COEF = 20                    # nonce: XOR-shared 20-bit string -> arithmetic
DABIT_AND_EQUIV = 2                         # offline cost of a daBit in AND-triple units

def per_attempt(opt):
    """returns dict op -> (ANDs, depth, note); opt: 'F_att' (hash inside) or 'ACH' (w1 public)"""
    ops = {}
    ops['nonce (B2A of XOR-shared 20-bit y)'] = (NL * B2A_DABITS_PER_COEF * DABIT_AND_EQUIV, 2)
    ops['w1 = HighBits(w): bitdec + 4 cmps'] = (NK * (BITDEC_AND + HIGHBITS_CMP * LOGQ), BITDEC_DEPTH + 4)
    if opt == 'F_att':
        ops['SHAKE256(mu||w1Encode) in MPC'] = (KECCAK_PERMS * KECCAK_AND_PER_PERM,
                                             KECCAK_PERMS * KECCAK_DEPTH_PER_PERM)
    ops['z-test: 1280 range cmps'] = (NL * CMP_AND, CMP_DEPTH)
    ops['r0-test: 1536 range cmps (x P_zpass)'] = (int(NK * CMP_AND * P_ZPASS), CMP_DEPTH)
    ops['openings (bit, w1, c~)'] = (0, 2)
    return ops

def bytes_h(b):
    for u in ('B', 'kB', 'MB', 'GB', 'TB'):
        if b < 1000: return f"{b:6.1f} {u}"
        b /= 1000
    return f"{b:.1f} PB"

def main():
    Ns = (8, 16, 50, 100)
    print("Per-attempt budget (ANDs, AND-depth rounds):")
    for opt in ('F_att', 'ACH'):
        ops = per_attempt(opt); tot = sum(a for a, _ in ops.values()); dep = sum(d for _, d in ops.values())
        print(f"\n  [{opt}]")
        for name, (a, d) in ops.items():
            print(f"    {name:44s} {a:9,d} ANDs  depth {d:4d}   ({100*a/tot:4.1f}%)")
        print(f"    {'TOTAL per attempt':44s} {tot:9,d} ANDs  depth {dep:4d}")
        print(f"    {'TOTAL per signature (x%.1f attempts)' % ATTEMPTS:44s} {int(tot*ATTEMPTS):9,d} ANDs  rounds {int(dep*ATTEMPTS)+3:4d}")
    print("\nPer-party communication per signature [EXTRAPOLATED]:")
    print("  online GMW: 2 bits/AND to each other party; offline authenticated AND triple:")
    print("  optimistic 20 B/AND/pair (silent-OT era), conservative 128 B/AND/pair (classic TinyOT);")
    print("  BMR: ~64 B/AND per garbler, times N garblers, per party.")
    hdr = f"  {'variant':6s} {'N':>4s} {'online':>10s} {'offline-opt':>12s} {'offline-cons':>13s} {'BMR-total':>11s}"
    print(hdr)
    for opt in ('F_att', 'ACH'):
        ands = sum(a for a, _ in per_attempt(opt).values()) * ATTEMPTS
        for N in Ns:
            online = ands * 2 * (N - 1) / 8
            off_o = ands * 20 * (N - 1)
            off_c = ands * 128 * (N - 1)
            bmr = ands * 64 * N
            print(f"  {opt:6s} {N:4d} {bytes_h(online):>10s} {bytes_h(off_o):>12s} {bytes_h(off_c):>13s} {bytes_h(bmr):>11s}")
    print(f"\n  (hash bytes {HASH_BYTES}, Keccak permutations {KECCAK_PERMS}, P[z-test pass] = {P_ZPASS:.3f})")

if __name__ == '__main__':
    main()
