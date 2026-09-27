"""
counts.py -- exact per-attempt preprocessing requirements of the benchmark circuits, as reported
by the MP-SPDZ 0.4.3 compiler ("Program requires at most"), written to bench/results/counts.json.

Compiled with K = 1 so every figure is per attempt. Needs the benchmark container (bench/dock.py).
"""
import json, os, re, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import dock

PROGRAMS = {"base": ("mldsa_offline", "mldsa_online"), "min": ("mldsa_offline_min", "mldsa_online_min"),
            "opt": ("mldsa_offline_opt", "mldsa_online_opt")}
RE_LINE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def requirements(prog):
    out = dock.sh(f"./compile.py {prog} 1 2>&1", timeout=3600).stdout
    block = out.split("Program requires at most:", 1)[1]
    req = {}
    for line in block.splitlines()[1:]:
        m = RE_LINE.match(line)
        if not m:
            break
        req[m.group(2)] = req.get(m.group(2), 0) + int(m.group(1))
    return req


def summarise(req):
    eda_bits = sum(v * int(re.search(r"length (\d+)", k).group(1))
                   for k, v in req.items() if "edabits of length" in k)
    return dict(bit_triples=req.get("bit triples", 0),
                edabit_bits=eda_bits,
                dabits=req.get("integer dabits", 0),
                integer_opens=req.get("integer opens", 0),
                and_equivalents=req.get("bit triples", 0) + eda_bits + req.get("integer dabits", 0),
                raw=req)


def main():
    dock.ensure_container()
    dock.sh("cp /bench/mpc/*.mpc Programs/Source/")
    res = {}
    for circ, (off, on) in PROGRAMS.items():
        res[circ] = {"w1_phase": summarise(requirements(off)), "test_phase": summarise(requirements(on))}
        tot = {k: res[circ]["w1_phase"][k] + res[circ]["test_phase"][k]
               for k in ("bit_triples", "edabit_bits", "dabits", "integer_opens", "and_equivalents")}
        res[circ]["total"] = tot
        print(circ, tot)
    res["source"] = "MP-SPDZ 0.4.3 compiler, 'Program requires at most', programs compiled with K = 1"
    with open(os.path.join(_here, "results", "counts.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
