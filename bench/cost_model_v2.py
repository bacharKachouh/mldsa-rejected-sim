"""
cost_model_v2.py -- cost of the open-w1 construction (scripts/threshold_openw1.py),
estimated figures, not measurements.  All [EXTRAPOLATED].

Differences from F_att (cost_model.py):
  * no SHAKE in MPC (hash public)                          -> depth 168/attempt gone
  * hint test in the clear after opening z (safe: public function of z, see the paper)
  * offline (message-independent): nonce B2A + carry/w1 per CONSUMED attempt
  * online: z-test + r0-test on shares for a WAVE of attempts in parallel
Two batching policies:
  ONE-SHOT K=24: test 24 attempts at once (1 wave), consume 24 attempts of preprocessing
  WAVES of 8:    test 8 at a time; E[waves] = 1/(1-0.8^8); unused attempts carry over
"""
import math
from cost_model import (NK, NL, CMP_AND, CMP_DEPTH, BITDEC_AND, BITDEC_DEPTH, HIGHBITS_CMP, LOGQ,
                        B2A_DABITS_PER_COEF, DABIT_AND_EQUIV, P_ZPASS, bytes_h)

P_ACC = 0.2                                  # per-attempt acceptance (all three tests), ~1/5
ONLINE_PER_ATTEMPT = NL * CMP_AND + int(NK * CMP_AND * P_ZPASS)      # z-test + r0-test (r0 only if z passes)
OFFLINE_PER_ATTEMPT = NL * B2A_DABITS_PER_COEF * DABIT_AND_EQUIV + NK * (BITDEC_AND + HIGHBITS_CMP * LOGQ)
D_GMW = BITDEC_DEPTH + 4                     # bit-decomposition + comparison + AND-tree ~ 10
D_GC = 2
DKG_ONE_TIME = (NL + NK) * 3 * DABIT_AND_EQUIV   # sample eta-bounded coefficients as shared bits

def policy(name, wave, K_consumed_per_sig, exp_waves):
    online = ONLINE_PER_ATTEMPT * wave * exp_waves
    offline = OFFLINE_PER_ATTEMPT * K_consumed_per_sig
    return dict(name=name, online=online, offline=offline, exp_waves=exp_waves)

pols = [policy("one-shot K=24", 24, 24, 1.0),
        policy("waves of 8", 8, 8 / (1 - 0.8 ** 8), 1 / (1 - 0.8 ** 8))]

def main():
    print("Per-signature AND budget [EXTRAPOLATED]:")
    print(f"  per attempt: online (z+r0 tests) {ONLINE_PER_ATTEMPT:,} ANDs; offline (nonce+w1) {OFFLINE_PER_ATTEMPT:,} ANDs")
    print(f"  one-time DKG: ~{DKG_ONE_TIME:,} AND-equivalents (negligible)")
    for p in pols:
        r_gmw = 1 + D_GMW * p['exp_waves'] + 1; r_gc = 1 + D_GC * p['exp_waves'] + 1
        print(f"  {p['name']:14s}: online {p['online']:,.0f} ANDs, offline {p['offline']:,.0f} ANDs, "
              f"E[online rounds] GMW {r_gmw:.1f} (min {2+D_GMW}), GC {r_gc:.1f} (min {2+D_GC})")
    print("\nPer-party communication per signature (|S| = N) [EXTRAPOLATED]:")
    print(f"  {'policy':14s} {'N':>4s} {'online GMW':>11s} {'offline opt':>12s} {'offline cons':>13s} {'GC online':>11s}")
    for p in pols:
        for N in (8, 16, 50, 100):
            on = p['online'] * 2 * (N - 1) / 8
            off_o = (p['online'] + p['offline']) * 20 * (N - 1)
            off_c = (p['online'] + p['offline']) * 128 * (N - 1)
            gc = p['online'] * 64 * N
            print(f"  {p['name']:14s} {N:4d} {bytes_h(on):>11s} {bytes_h(off_o):>12s} {bytes_h(off_c):>13s} {bytes_h(gc):>11s}")
    print("\n  'offline' includes the triples for the online tests (consumed per signature) and the")
    print("  message-independent nonce+w1 computation; 'GC online' = garbled tables for the test circuit")
    print("  (constant-round option, replaces 'online GMW'; its own triples are in 'offline').")

if __name__ == '__main__':
    main()
