"""
table2.py -- every figure of the paper's cost section and Table 2, from bench/results/.

  * gate counts per attempt (baseline and optimised circuits)      counts.json, COUNTED
  * online traffic per attempt of a non-coordinating party          result files, MEASURED
  * online rounds per batch of the test phase, N = 2, 4, 8          result files, MEASURED
  * semi-honest estimate per party per signature, N = 2, 4, 8       ESTIMATED: measured online traffic
      (N = 2, assumed constant in N) plus FOLEAGE's communication formula for F2 triples,
      2(N-1) * C / 3^18 bits per triple with C = 13 MB per 3^18 correlations, plus one broadcast bit
      when N > 2, applied to the counted AND-equivalents of the optimised circuit
  * malicious cost per signature at N = 2 along the optimisation steps, and the steps not kept

Attempts per signature: batches of K attempts until one accepts, acceptance 0.2 per attempt, so
K / (1 - 0.8^K) attempts are consumed (6.8 for K = 4, 5.0 for K = 1).
Run:  python bench/table2.py
"""
import glob, json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
P_ACCEPT = 0.2


def attempts_per_signature(K):
    return K / (1 - (1 - P_ACCEPT) ** K)


def runs():
    out = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        if p.endswith("counts.json"):
            continue
        r = json.load(open(p))
        if not r.get("verified"):
            continue
        c = r["config"]
        out.setdefault((c.get("circuit", "base"), c["protocol"], c["N"], c["K"]), []).append(r)
    return out


def per_attempt_total(r):          # party 0, everything (preprocessing + online), MB per attempt
    return sum(b[ph]["party0_MB"] for b in r["batches"] for ph in ("offline", "online")) / r["attempts"]


def per_attempt_online_p1(r):      # a party that does not coordinate: party 1, online only
    return sum(b[ph]["party1"]["online"]["MB"] for b in r["batches"] for ph in ("offline", "online")
               if "party1" in b[ph]) / r["attempts"]


def test_rounds(r):
    return statistics.median(b["online"]["online"]["rounds"] for b in r["batches"])


def med(xs):
    return statistics.median(xs) if xs else None


def main():
    counts = json.load(open(os.path.join(RES, "counts.json")))
    R = runs()
    base, opt = counts["base"]["total"]["and_equivalents"], counts["opt"]["total"]["and_equivalents"]
    print(f"AND-equivalents per attempt: baseline {base:,}, optimised {opt:,}  (COUNTED)")

    on = med([per_attempt_online_p1(r) for r in R.get(("opt", "semi", 2, 4), [])])
    print(f"online traffic per attempt, non-coordinating party, optimised circuit, N = 2: {on:.3f} MB")
    for N in (2, 4, 8):
        rs = R.get(("min", "semi", N, 4), [])
        p1 = [per_attempt_online_p1(r) for r in rs if any("party1" in b["online"] for b in r["batches"])]
        print(f"  unoptimised circuit, N = {N}: online per attempt (party 1) "
              f"{med(p1):.3f} MB, test-phase online rounds per batch {med([test_rounds(r) for r in rs]):.0f}")
    print(f"optimised circuit, N = 2: test-phase online rounds per batch "
          f"{med([test_rounds(r) for r in R.get(('opt', 'semi', 2, 4), [])]):.0f}")

    A = attempts_per_signature(4)
    print(f"\nsemi-honest estimate per party per signature ({A:.1f} attempts, ESTIMATED):")
    for N in (2, 4, 8):
        bits = 2 * (N - 1) * 13e6 * 8 / 3 ** 18 + (1 if N > 2 else 0)
        pre = opt * bits / 8 / 1e6
        print(f"  N = {N}: online {on * A:.2f} MB + preprocessing {pre * A:.2f} MB = {(on + pre) * A:.2f} MB")

    print("\nmalicious security, N = 2, per party per signature (MEASURED):")
    steps = [("untuned circuit, MASCOT, batch 4", "base", "mascot", 4),
             ("minimal circuit, MASCOT, batch 4", "min", "mascot", 4),
             ("minimal circuit, LowGear, batch 4", "min", "lowgear", 4),
             ("minimal circuit, LowGear, batch 1", "min", "lowgear", 1),
             ("optimised circuit, LowGear, batch 4", "opt", "lowgear", 4),
             ("optimised circuit, LowGear, batch 1", "opt", "lowgear", 1)]
    first = None
    for name, circ, proto, K in steps:
        v = med([per_attempt_total(r) for r in R.get((circ, proto, 2, K), [])])
        sig = v * attempts_per_signature(K) / 1000
        first = first or sig
        print(f"  {name:38s} {sig:8.1f} GB   ({first / sig:.1f}x below the untuned circuit)")
    print("tried and not kept (per signature):")
    for name, circ, proto, K in [("covert, CowGear, optimised, batch 1", "opt", "cowgear", 1),
                                 ("HighGear, minimal, batch 4", "min", "highgear", 4),
                                 ("SPDZ2k, minimal, batch 4", "min", "spdz2k", 4)]:
        v = med([per_attempt_total(r) for r in R.get((circ, proto, 2, K), [])])
        print(f"  {name:38s} {v * attempts_per_signature(K) / 1000:8.1f} GB")


if __name__ == "__main__":
    main()
