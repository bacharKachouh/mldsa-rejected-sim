"""
08 -- s = 3 certificate (paper, Lemma "s = 3 certificate"), certified: no floating point enters
the bound.  The search is integer arithmetic; every full survivor is evaluated exactly as the
rational (64^6 / P)^42 with P = prod over the six coordinates of (x, Cx) of max(64, |v|_c); parts (i)
and (ii) are the rational bounds of rigorous.py.  Each chunk writes its worst exact value of (iii)
to s3_chunk_<a>_<b>.json; `combine` reads them and prints the certified bound.

For every 3-set S = {i1<i2<i3} of slots, characters t_j = sum c_i z_i^j; a block x = (t_j, t_{j+1}, t_{j+2})
determines t and the next block is C x with C = M^3, M the companion matrix of prod (X - z_i).
Two-block Holder with L/2 = 42:   sum_{t != 0} B(t) <= sum_{x != 0} (Phi(x) Phi(Cx))^{42}.
Split x in Z_q^3 minus 0:
  (i)   max|x_j| > X:            total <= 3 * (2 * 64^42 X^{-41} / 41) * Lambda(42)^2   [tail bound]
  (ii)  max|x_j| <= X and max|(Cx)_j| > X':  each term <= (64/(X'+1))^{42}, count <= (2X+1)^3
  (iii) max|x_j| <= X and max|(Cx)_j| <= X': found by meet-in-the-middle and evaluated exactly.
With X = 128 the exact tail (i) is 2^-23.94 (summed exactly by 06/07; the integral bound gives 2^-23.70):
  (ii) <= 257^3 * (64/513)^42 = 2^24 * 2^-126 = 2^-102.
The certificate for S is  (iii) + 2^-23.94 + 2^-102.1.  Result (all 2,763,520 sets, 4 chunks x ~4.6 h):
  worst (iii) = 2^-24.2 at the 3-element cosets {i, i+64, i+128}; hence Sigma_S <= 2^-23.06 for every S.
Meet in the middle: enumerate (x1,x2) in [-X,X]^2 (66049 pairs), p = x1*C[:,0] + x2*C[:,1] mod q;
for coordinate 0, x3 must satisfy (p_0 + x3*C[0,2]) mod q in [-X',X']; precompute the sorted
array of x3*C[0,2] mod q for x3 in [-X,X]; a candidate exists iff some element lies in the arc
[-X'-p_0, X'-p_0] mod q  -> two searchsorted calls per pair.  Survivors (expected ~ 66049 * 193 * 1025 / q
~ 1600 per set) are checked on coordinates 1 and 2 exactly, and the rare full survivors are evaluated.
Run:  python scripts/08_s3_certificate.py <start_i1> <end_i1>     one chunk
      python scripts/08_s3_certificate.py splits <k>                  print k balanced chunks
      python scripts/08_s3_certificate.py combine                     certified bound from all chunks
"""
import json, math, sys, time
from fractions import Fraction
import numpy as np
from params import q, n

zeta = 1753
def root(i): return pow(zeta, 2 * int(i) + 1, q)
X, XP, LH = 128, 512, 42

def phiL(v):
    vm = v % q; vc = np.minimum(vm, q - vm).astype(np.float64); vc = np.maximum(vc, 1.0)
    return np.minimum(1.0, 64.0 / vc) ** LH

def main(i_start=0, i_end=n):
    t0 = time.time()
    r = np.arange(-X, X + 1, dtype=np.int64)
    x1, x2 = np.meshgrid(r, r, indexing='ij'); x1 = x1.ravel(); x2 = x2.ravel()
    x3s = r.copy()
    base12 = phiL(x1) * phiL(x2)
    roots = [root(i) for i in range(n)]
    worst = 0.0; worst_S = None; nsets = 0; nfull = 0
    for i1 in range(i_start, i_end):
        for i2 in range(i1 + 1, n):
            for i3 in range(i2 + 1, n):
                z1, z2, z3 = roots[i1], roots[i2], roots[i3]
                # monic poly X^3 - e1 X^2 + e2 X - e3 ; recurrence t_{j+3} = e1 t_{j+2} - e2 t_{j+1} + e3 t_j
                e1 = (z1 + z2 + z3) % q; e2 = (z1 * z2 + z1 * z3 + z2 * z3) % q; e3 = (z1 * z2 * z3) % q
                # companion M on (t_j,t_{j+1},t_{j+2}): rows [0,1,0],[0,0,1],[e3,-e2,e1]; C = M^3
                M = np.array([[0, 1, 0], [0, 0, 1], [e3, (-e2) % q, e1]], dtype=object)
                C = M.dot(M) % q; C = C.dot(M) % q
                C = np.array([[int(v) for v in row] for row in C], dtype=np.int64)
                # partial p = x1*C[:,0] + x2*C[:,1]  (3 coords), exact in int64 (< 2^23*2^8*2)
                p0 = (x1 * C[0, 0] + x2 * C[0, 1]) % q
                # coordinate-0 test: need x3 with (p0 + x3*C[0,2]) mod q in [-XP, XP]
                tab = np.sort((x3s * C[0, 2]) % q)
                lo = (-XP - p0) % q; hi = (XP - p0) % q          # arc [lo, hi] mod q (length 2XP+1 < q)
                # count elements of tab in arc: handle wrap
                a = np.searchsorted(tab, lo, side='left'); b = np.searchsorted(tab, hi, side='right')
                nowrap = lo <= hi
                cnt = np.where(nowrap, b - a, (len(tab) - a) + b)
                cand = np.flatnonzero(cnt > 0)
                val = 0.0
                if len(cand):
                    # exact check for candidates on all coordinates: enumerate x3 for each candidate pair
                    cx1 = x1[cand]; cx2 = x2[cand]
                    # broadcast over x3
                    P = np.stack([(cx1 * C[k, 0] + cx2 * C[k, 1]) % q for k in range(3)])   # 3 x m
                    Y = (P[:, :, None] + (x3s[None, None, :] * C[:, 2][:, None, None])) % q   # 3 x m x 257
                    Yc = np.minimum(Y, q - Y)
                    ok = (Yc <= XP).all(axis=0)                                           # m x 257
                    if ok.any():
                        m_idx, x3_idx = np.nonzero(ok)
                        xx1 = cx1[m_idx]; xx2 = cx2[m_idx]; xx3 = x3s[x3_idx]
                        nz = (xx1 != 0) | (xx2 != 0) | (xx3 != 0)
                        if nz.any():
                            xx1, xx2, xx3, m_idx, x3_idx = xx1[nz], xx2[nz], xx3[nz], m_idx[nz], x3_idx[nz]
                            yy = Y[:, m_idx, x3_idx]
                            def f(v):
                                v = int(v) % q; return max(64, min(v, q - v))
                            exact = Fraction(0)
                            for u in range(len(xx1)):
                                P = (f(xx1[u]) * f(xx2[u]) * f(xx3[u])
                                     * f(yy[0][u]) * f(yy[1][u]) * f(yy[2][u]))
                                exact += Fraction(2 ** (36 * LH), P ** LH)
                            nfull += len(xx1)
                            val = exact
                nsets += 1
                if val > worst: worst, worst_S = Fraction(val), (i1, i2, i3)
        if (i1 - i_start) % 8 == 0:
            print(f"   i1={i1}: sets so far {nsets}, worst exact part {worst:.3e} at {worst_S}, full survivors {nfull}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"DONE sets={nsets}  worst exact part (iii) = 2^{math.log2(worst) if worst > 0 else float('-inf'):.2f} "
          f"at S={worst_S}; full survivors {nfull}  [{time.time()-t0:.0f}s]")
    out = {"start": i_start, "end": i_end, "sets": nsets, "full_survivors": nfull,
           "worst_S": worst_S, "worst_num": str(Fraction(worst).numerator),
           "worst_den": str(Fraction(worst).denominator)}
    with open(f"s3_chunk_{i_start}_{i_end}.json", "w") as fh:
        json.dump(out, fh)

def splits(k):
    """k chunks of first-slot indices with roughly equal numbers of triples."""
    w = [(n - 1 - i) * (n - 2 - i) // 2 for i in range(n)]
    total = sum(w); bounds = [0]; acc = 0
    for i in range(n):
        acc += w[i]
        if acc >= total * len(bounds) / k and len(bounds) < k:
            bounds.append(i + 1)
    bounds.append(n)
    return list(zip(bounds[:-1], bounds[1:]))

def combine():
    import glob
    from rigorous import Lambda_up, tail_up, log2_up, fmt
    files = sorted(glob.glob("s3_chunk_*.json"))
    chunks = [json.load(open(f)) for f in files]
    covered = sorted((c["start"], c["end"]) for c in chunks)
    ok = covered[0][0] == 0 and covered[-1][1] == n and all(a[1] == b[0] for a, b in zip(covered, covered[1:]))
    nsets = sum(c["sets"] for c in chunks)
    worst = max(chunks, key=lambda c: Fraction(int(c["worst_num"]), int(c["worst_den"])))
    iii = Fraction(int(worst["worst_num"]), int(worst["worst_den"]))
    part_i = 3 * tail_up(LH) * Lambda_up(LH) ** 2
    part_ii = Fraction(257 ** 3 * 64 ** LH, 513 ** LH)
    total = iii + part_i + part_ii
    print(f"chunks {len(chunks)}, contiguous cover of 0..{n}: {ok}; sets {nsets} (expected {n*(n-1)*(n-2)//6})")
    print(f"(iii) worst exact value <= {fmt(log2_up(iii))} at S={tuple(worst['worst_S'])}")
    print(f"(i)   <= {fmt(log2_up(part_i))}   (ii) <= {fmt(log2_up(part_ii))}")
    print(f"CERTIFIED: Sigma_S <= {fmt(log2_up(total))} for every three-element slot set S"
          if ok and nsets == n * (n - 1) * (n - 2) // 6 else "INCOMPLETE: not all sets covered")

if __name__ == "__main__":
    if sys.argv[1:2] == ["splits"]:
        print(" ".join(f"{a}:{b}" for a, b in splits(int(sys.argv[2]))))
    elif sys.argv[1:2] == ["combine"]:
        combine()
    else:
        a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        b = int(sys.argv[2]) if len(sys.argv) > 2 else n
        main(a, b)
