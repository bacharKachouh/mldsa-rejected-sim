r"""
s3_check.py -- independent re-implementation of the s = 3 certificate search (Lemma "s = 3
certificate", part (iii)). No repo imports; shares no code with scripts/08_s3_certificate.py.

For a slot set S = {i1 < i2 < i3} with roots w_i = 1753^(2i+1) mod q, a character is
t_j = sum_i c_i w_i^j. With V = (w_i^m)_{m<3, i in S} and D = diag(w_i), consecutive blocks of three
coordinates satisfy t|_{J_{m+1}} = C t|_{J_m} with C = V D^3 V^{-1} mod q. This program forms C
that way (08 uses the cube of the companion matrix) and finds every x in [-128,128]^3 \ 0 with
max_k |(Cx)_k|_c <= 512 ("full survivors", the terms of part (iii)).

Meet in the middle on the LAST row of C, from the other side than 08: the 66049 partial sums
p = C[2,1] x2 + C[2,2] x3 mod q are sorted, and for each of the 257 values of x1 the pairs with
(C[2,0] x1 + p) mod q in [-512, 512] are read off by binary search. Candidates are checked on all
three rows in integer arithmetic.

Outputs
  results/s3_survivors.txt   the witness: one line "i1 i2 i3 x1 x2 x3" per full survivor
Checks
  * per-chunk survivor counts equal the "full_survivors" fields of results/s3_chunks/*.json;
  * the largest per-set exact value sum_x (2^36 / P(x))^42, P(x) = prod over the six coordinates
    of (x, Cx) of max(64, |v|_c), equals the "worst" rational stored by 08, at the same set.

Run:  python s3_check.py [workers]        (about 3 CPU-hours; minutes on a many-core machine)
      python s3_check.py --witness        (re-check the witness file only: seconds)
"""
import glob, json, math, os, sys, time
from fractions import Fraction
from multiprocessing import Pool
import numpy as np

q, n, zeta = 8380417, 256, 1753
X, XP, LH = 128, 512, 42
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
W = [pow(zeta, 2 * i + 1, q) for i in range(n)]


def cmat(S):
    """C = V D^3 V^{-1} mod q as a 3x3 list of Python ints."""
    V = [[pow(W[i], m, q) for i in S] for m in range(3)]
    # inverse of the 3x3 matrix V by the adjugate
    a, b, c = V[0]; d, e, f = V[1]; g, h, k = V[2]
    adj = [[e * k - f * h, c * h - b * k, b * f - c * e],
           [f * g - d * k, a * k - c * g, c * d - a * f],
           [d * h - e * g, b * g - a * h, a * e - b * d]]
    det = (a * adj[0][0] + b * adj[1][0] + c * adj[2][0]) % q
    di = pow(det, q - 2, q)
    Vi = [[adj[r][s] * di % q for s in range(3)] for r in range(3)]
    VD = [[V[r][s] * pow(W[S[s]], 3, q) % q for s in range(3)] for r in range(3)]
    return [[sum(VD[r][t] * Vi[t][s] for t in range(3)) % q for s in range(3)] for r in range(3)]


def cabs(v):
    v %= q
    return min(v, q - v)


R = np.arange(-X, X + 1, dtype=np.int64)
G2, G3 = [g.ravel() for g in np.meshgrid(R, R, indexing="ij")]


def survivors(S):
    C = cmat(S)
    p = (C[2][1] * G2 + C[2][2] * G3) % q
    order = np.argsort(p, kind="stable"); ps = p[order]
    out = []
    for x1 in range(-X, X + 1):
        lo = (-XP - C[2][0] * x1) % q
        hi = (XP - C[2][0] * x1) % q
        if lo <= hi:
            idx = order[np.searchsorted(ps, lo, "left"):np.searchsorted(ps, hi, "right")]
        else:
            idx = np.concatenate((order[np.searchsorted(ps, lo, "left"):],
                                  order[:np.searchsorted(ps, hi, "right")]))
        for j in idx.tolist():
            x = (x1, int(G2[j]), int(G3[j]))
            if x == (0, 0, 0):
                continue
            if all(cabs(C[r][0] * x[0] + C[r][1] * x[1] + C[r][2] * x[2]) <= XP for r in range(3)):
                out.append(x)
    return out


def work_i1(i1):
    t0 = time.time(); found = []; nsets = 0
    for i2 in range(i1 + 1, n):
        for i3 in range(i2 + 1, n):
            nsets += 1
            for x in survivors((i1, i2, i3)):
                found.append((i1, i2, i3) + x)
    return i1, nsets, found, time.time() - t0


def value(S, xs):
    """exact sum over the survivors xs of (2^36 / P(x))^42."""
    C = cmat(S); tot = Fraction(0)
    for x in xs:
        cx = [(C[r][0] * x[0] + C[r][1] * x[1] + C[r][2] * x[2]) for r in range(3)]
        P = 1
        for v in list(x) + cx:
            P *= max(64, cabs(v))
        tot += Fraction(2 ** (36 * LH), P ** LH)
    return tot


def check(rows, nsets_total):
    chunks = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "s3_chunks", "*.json")))]
    ok = True
    for ch in sorted(chunks, key=lambda c: c["start"]):
        mine = sum(1 for r in rows if ch["start"] <= r[0] < ch["end"])
        flag = "ok" if mine == ch["full_survivors"] else "MISMATCH"
        ok &= mine == ch["full_survivors"]
        print(f"   i1 in [{ch['start']:3d},{ch['end']:3d}): full survivors {mine:4d} (08: {ch['full_survivors']:4d}) {flag}")
    bys = {}
    for r in rows:
        bys.setdefault(tuple(r[:3]), []).append(tuple(r[3:]))
    vals = {S: value(S, xs) for S, xs in bys.items()}
    Sw = max(vals, key=vals.get); vw = vals[Sw]
    w08 = max(chunks, key=lambda c: Fraction(int(c["worst_num"]), int(c["worst_den"])))
    v08 = Fraction(int(w08["worst_num"]), int(w08["worst_den"]))
    tied = sorted(S for S, v in vals.items() if v == vw)
    same = vw == v08 and tuple(w08["worst_S"]) in tied
    ok &= same
    print(f"sets searched {nsets_total} (expected {n*(n-1)*(n-2)//6}); sets with a survivor {len(bys)}; survivors {len(rows)}")
    print(f"largest (iii): log2 = {math.log2(vw.numerator) - math.log2(vw.denominator):.4f}, "
          f"exactly equal to 08's stored rational: {vw == v08}; attained at {len(tied)} sets {tied}")
    print("PASS" if ok and nsets_total in (None, n * (n - 1) * (n - 2) // 6) else "FAIL")


if __name__ == "__main__":
    path = os.path.join(RES, "s3_survivors.txt")
    if sys.argv[1:2] == ["--witness"]:
        rows = [tuple(map(int, line.split())) for line in open(path) if line.strip() and line[0] != "#"]
        # re-verify each witness row: x in the box, x != 0, Cx in the box
        for r in rows:
            S, x = r[:3], r[3:]
            C = cmat(S)
            assert x != (0, 0, 0) and max(map(abs, x)) <= X
            assert all(cabs(C[k][0] * x[0] + C[k][1] * x[1] + C[k][2] * x[2]) <= XP for k in range(3))
        print(f"witness rows verified: {len(rows)}")
        check(rows, None)
        sys.exit()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    t0 = time.time(); rows = []; nsets = 0
    # largest first so that the pool stays busy
    with Pool(workers) as pool:
        for i1, ns, found, dt in pool.imap_unordered(work_i1, range(n - 2), chunksize=1):
            rows += found; nsets += ns
            print(f"   i1={i1:3d}: {ns:6d} sets, {len(found)} survivors [{dt:.0f}s; total {time.time()-t0:.0f}s]", flush=True)
    rows.sort()
    with open(path, "w") as fh:
        fh.write("# s = 3 certificate, part (iii): every x in [-128,128]^3 \\ 0 with max|(Cx)_k|_c <= 512\n")
        fh.write("# columns: i1 i2 i3 x1 x2 x3   (slots in the labelling w_i = 1753^(2i+1) mod q)\n")
        for r in rows:
            fh.write(" ".join(map(str, r)) + "\n")
    check(rows, nsets)
    print(f"[{time.time()-t0:.0f}s wall]")
