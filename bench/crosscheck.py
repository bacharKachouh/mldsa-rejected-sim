"""
crosscheck.py -- correctness gate for the MPC programs (debug builds, N = 2, semi-honest for speed).

For many seeds: the MPC's w1 and r0 must equal FIPS Decompose of A*y (y revealed in debug builds
only), its j* must equal the FIPS-exact reference bench/mldsa_host.fips_first_accepting, and the
signature must verify. Separately reports how often scripts/threshold_openw1.Ideal.first_accepting,
whose hint test is makehint(ct0 - cs2, w - cs2) instead of FIPS MakeHint(-ct0, w - cs2 + ct0),
disagrees with the FIPS reference on the same batches.
"""
import os, sys
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import mldsa_host as mh
from run import Bench
from core import q, l, n, highbits


def ideal_disagreements(trials=3000, seed=0):
    """FIPS reference vs the repo's Ideal hint test, on random attempts (no MPC needed)."""
    from threshold_openw1 import Ideal
    key = mh.keygen(seed)
    ideal = Ideal(np.random.default_rng(0), 2, 2)
    ideal.s1, ideal.s2, ideal.t0 = key['s1'], key['s2'], key['t0']
    rng = np.random.default_rng(seed + 1)
    diff = 0
    for i in range(trials):
        y = rng.integers(-(1 << 19) + 1, (1 << 19) + 1, size=(l, n), dtype=np.int64)
        w = mh.matvec_int(key['A'], y) % q
        c = mh.challenge_for(b"x%d" % i, highbits(w))
        a, _ = mh.fips_first_accepting(key, [y], [w], [c])
        b, _ = ideal.first_accepting(b"", [dict(y=y, w=w)], [c])
        diff += (a is None) != (b is None)
    return diff, trials


def main():
    circuit = sys.argv[1] if len(sys.argv) > 1 else 'base'
    protocol = sys.argv[2] if len(sys.argv) > 2 else 'semi'
    seeds_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seeds = range(1, seeds_n + 1)
    bad = 0
    total_batches = total_w1 = 0
    for K in (1, 4):
        bench = Bench(2, protocol, K, True, 'local', circuit)
        for s in seeds:
            key = mh.keygen(s)
            bench.setup(key)
            batches, ok, checks = bench.sign(key, f"cross-{s}".encode())
            total_batches += len(batches)
            total_w1 += checks['w1_checked']
            good = ok and checks['w1_matches'] == checks['w1_checked'] and checks['jstar_matches_ref']
            bad += not good
            print(f"[{circuit}/{protocol}] K={K} seed={s}: batches={len(batches)} verified={ok} "
                  f"w1 {checks['w1_matches']}/{checks['w1_checked']} j*=ref {checks['jstar_matches_ref']}")
    print(f"MPC vs FIPS reference: {total_batches} batches, {total_w1} attempts, failures: {bad}")
    d, t = ideal_disagreements()
    print(f"Ideal.first_accepting vs FIPS reference: {d} of {t} random attempts disagree on accept/reject")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
