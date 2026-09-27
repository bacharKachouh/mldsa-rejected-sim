"""
run.py -- benchmark driver for the open-w1 threshold ML-DSA functionalities in MP-SPDZ.

One run = produce one signature with batch size K:
  setup (key load, one-time) -> [offline(K): F_nonce+F_w1, reveals w1 -> host hash ->
  online(K): F_test, reveals j*, z] repeated until an attempt accepts -> host hint + stock verify.
A run's numbers are recorded only if the signature verifies under core.verify.

Every program runs with MP-SPDZ's -v, which reports for the same run the cost of the online
phase and of the preprocessing phase separately (MB sent, rounds, seconds, party 0); both are
recorded per program under 'online' and 'prep', alongside the totals.

Usage: python bench/run.py --N 2 --protocol mascot --K 4 [--net wan] [--debug] [--seed 1]
"""
import argparse, datetime, json, os, re, subprocess, sys, time
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import dock
import mldsa_host as mh
from core import q, n, k, l, highbits

SCRIPTS = {'mascot': 'mascot.sh', 'semi': 'semi.sh', 'spdz2k': 'spdz2k.sh', 'lowgear': 'lowgear.sh',
           # honest majority (t < N/2), Quorus's setting: Shamir sharing, semi-honest / malicious
           'shamir': 'shamir.sh', 'mal-shamir': 'mal-shamir.sh',
           # covert (CowGear: LowGear with covert security) and HighGear (HE preprocessing, SHE-heavier)
           'cowgear': 'cowgear.sh', 'highgear': 'highgear.sh'}
RING = {'spdz2k': 128}     # ring protocols: compile and run mod 2^128 (A*y reduction needs ~85-bit products)
RESULTS = os.path.join(_here, "results")

RE_TIME = re.compile(r"^Time = ([\d.e+-]+) seconds", re.M)
RE_SENT = re.compile(r"^Data sent = ([\d.e+-]+) MB in ~(\d+) rounds", re.M)
RE_GLOBAL = re.compile(r"^Global data sent = ([\d.e+-]+) MB", re.M)
RE_SPLIT = re.compile(r"Spent ([\d.e+-]+) seconds \(([\d.e+-]+) MB, (\d+) rounds.*?\) on the online phase "
                      r"and ([\d.e+-]+) seconds \(([\d.e+-]+) MB, (\d+) rounds.*?\) on the preprocessing", re.M)


def parse_stats(out):
    t, s, g = RE_TIME.findall(out), RE_SENT.findall(out), RE_GLOBAL.findall(out)
    if not (t and s and g):
        raise RuntimeError("MP-SPDZ statistics not found in output:\n" + out[-3000:])
    st = dict(time_s=float(t[-1]), party0_MB=float(s[-1][0]), rounds=int(s[-1][1]),
              global_MB=float(g[-1]), online=None, prep=None)
    sp = RE_SPLIT.findall(out)
    if sp:                        # absent only for programs that consume no preprocessing (setup)
        on_s, on_mb, on_r, pre_s, pre_mb, pre_r = sp[-1]
        st['online'] = dict(time_s=float(on_s), MB=float(on_mb), rounds=int(on_r))
        st['prep'] = dict(time_s=float(pre_s), MB=float(pre_mb), rounds=int(pre_r))
    return st


def centered(x, m=q):
    x = np.asarray(x, dtype=np.int64) % m
    return np.where(x > m // 2, x - m, x)


def ints(line):
    """Integers of an output line after its tag (e.g. 'W1 0 3 [..]' -> [0, 3, ..])."""
    return [int(v) for v in re.findall(r"-?\d+", line.split(None, 1)[1])]


class Bench:
    def __init__(self, N, protocol, K, debug, net, circuit='base'):
        self.N, self.protocol, self.K, self.debug, self.net = N, protocol, K, debug, net
        suffix = '' if circuit == 'base' else '_' + circuit
        dock.ensure_container()
        dock.sh("cp /bench/mpc/*.mpc Programs/Source/")
        self.p_setup = self._compile("mldsa_setup")
        self.p_off = self._compile("mldsa_offline" + suffix, str(K), *(["debug"] if debug else []))
        self.p_on = self._compile("mldsa_online" + suffix, str(K))

    def _compile(self, name, *args):
        ring = f"-R {RING[self.protocol]} " if self.protocol in RING else ""
        dock.sh(f"./compile.py {ring}{name} {' '.join(args)} > /dev/null", timeout=3600)
        return "-".join([name, *args])

    def _run(self, prog):
        ring = f" -R {RING[self.protocol]}" if self.protocol in RING else ""
        cmd = f"PLAYERS={self.N} Scripts/{SCRIPTS[self.protocol]} {prog} -v{ring} 2>&1"
        t0 = time.time()
        r = dock.sh(cmd)
        st = parse_stats(r.stdout)
        st['wall_s'] = time.time() - t0
        # party 1 is not the coordinator of MP-SPDZ's star-shaped openings (party 0 is)
        log1 = dock.sh(f"cat logs/{prog}-1", check=False).stdout
        st['party1'] = parse_stats(log1.replace("party 1 only", "party 0 only")) if "Data sent" in log1 else None
        return r.stdout, st

    def setup(self, key):
        data = " ".join(str(int(v)) for v in np.r_[key['s1'].ravel(), key['s2'].ravel()])
        for p in range(self.N):
            dock.put_text(f"Player-Data/Input-P{p}-0", data + "\n" if p == 0 else "0\n")
        dock.sh("rm -f Persistence/Transactions-P*.data")
        out, st = self._run(self.p_setup)
        if "SETUP done" not in out:
            raise RuntimeError("setup failed:\n" + out[-2000:])
        return st

    def sign(self, key, msg):
        A_pub = " ".join(str(v) for v in centered(key['A']).ravel())
        dock.put_text(f"Programs/Public-Input/{self.p_off}", A_pub + "\n")
        batches, checks = [], dict(w1_matches=0, w1_checked=0, jstar_matches_ref=None)
        while True:
            out_off, st_off = self._run(self.p_off)
            w1 = np.zeros((self.K, k, n), dtype=np.int64)
            for line in out_off.splitlines():
                if line.startswith("W1 "):
                    v = ints(line); w1[v[0], v[1]] = v[2:]
            cs = [mh.challenge_for(msg, w1[j]) for j in range(self.K)]
            pub = []
            for j in range(self.K):
                ct0 = mh.polymul_vec(cs[j], key['t0'])
                hi, lo = mh.hint_thresholds(w1[j], ct0)
                pub += list(cs[j]) + list(hi.ravel()) + list(lo.ravel())
            dock.put_text(f"Programs/Public-Input/{self.p_on}", " ".join(map(str, pub)) + "\n")
            out_on, st_on = self._run(self.p_on)
            jstar = ints([ln for ln in out_on.splitlines() if ln.startswith("JSTAR")][0])[0]
            batches.append(dict(offline=st_off, online=st_on, jstar=jstar))
            if self.debug:
                self._crosscheck(key, out_off, w1, cs, jstar, checks)
            if jstar >= 0:
                z = np.array(ints([ln for ln in out_on.splitlines() if ln.startswith("Z ")][0]),
                             dtype=np.int64).reshape(l, n)
                sig, ok = mh.assemble_and_verify(key, msg, cs[jstar], z)
                return batches, ok, checks

    def _crosscheck(self, key, out_off, w1, cs, jstar, checks):
        ys = np.zeros((self.K, l, n), dtype=np.int64)
        r0 = np.zeros((self.K, k, n), dtype=np.int64)
        for line in out_off.splitlines():
            if line.startswith("Y "):
                v = ints(line); ys[v[0]] = np.array(v[1:]).reshape(l, n)
            elif line.startswith("R "):
                v = ints(line); r0[v[0], v[1]] = v[2:]
        ws = [mh.matvec_int(key['A'], ys[j]) % q for j in range(self.K)]
        for j in range(self.K):
            r1_ref, r0_ref = mh.decompose(ws[j])
            checks['w1_checked'] += 1
            checks['w1_matches'] += int(np.array_equal(w1[j], r1_ref) and np.array_equal(r0[j], r0_ref))
        ref_j, _ = mh.fips_first_accepting(key, list(ys), ws, cs)
        ref_j = -1 if ref_j is None else ref_j
        ok = (ref_j == jstar)
        checks['jstar_matches_ref'] = ok if checks['jstar_matches_ref'] is None else (checks['jstar_matches_ref'] and ok)
        checks.setdefault('batches', []).append(dict(mpc=jstar, ref=ref_j))


def git_commit():
    r = subprocess.run(["git", "-C", _here, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def set_net(net):
    if net == 'wan':
        dock.sh("tc qdisc replace dev lo root netem delay 25ms rate 1gbit")
    else:
        dock.sh("tc qdisc del dev lo root 2>/dev/null || true", check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True, choices=range(2, 17))
    ap.add_argument("--protocol", default="mascot", choices=list(SCRIPTS))
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--net", default="local", choices=["local", "wan"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--circuit", default="base", choices=["base", "min", "opt"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    key = mh.keygen(a.seed)
    msg = f"bench-{a.seed}".encode()
    b = Bench(a.N, a.protocol, a.K, a.debug, a.net, a.circuit)
    set_net(a.net)
    try:
        setup_stats = b.setup(key)
        batches, ok, checks = b.sign(key, msg)
    finally:
        set_net('local')
    rec = dict(config=vars(a), when=datetime.datetime.now().isoformat(timespec="seconds"),
               git=git_commit(), mpspdz="0.4.3", setup=setup_stats, batches=batches,
               attempts=a.K * len(batches), verified=ok, checks=checks)
    print(json.dumps(dict(verified=ok, batches=len(batches), attempts=rec['attempts'],
                          checks={k_: v for k_, v in checks.items() if k_ != 'batches'}), indent=None))
    if not ok:
        print("SIGNATURE DID NOT VERIFY - no record written", file=sys.stderr)
        sys.exit(1)
    if not a.debug:
        os.makedirs(RESULTS, exist_ok=True)
        name = f"{rec['when'].replace(':', '')}-N{a.N}-{a.protocol}-K{a.K}-{a.net}-{a.circuit}-s{a.seed}{a.tag}.json"
        with open(os.path.join(RESULTS, name), "w") as f:
            json.dump(rec, f, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    main()
