"""
17 -- SampleInBall is exactly uniform when its stream is (Lemma "SampleInBall").

FIPS 204, Algorithm 29, with the stream replaced by its values: the first tau bits h_0..h_{tau-1}
of the first eight bytes give the signs, and for i = n - tau, ..., n - 1 the accepted byte j is
uniform on {0, ..., i} (a uniform byte is rejected while it exceeds i). Then
    c_i <- c_j ;  c_j <- (-1)^{h[i + tau - n]}.
For small (n, tau) the program enumerates every sequence (j_{n-tau}, ..., j_{n-1}) with
j_i in {0..i} and every sign vector, runs the loop, and checks that every element of B_tau
(exactly tau coefficients in {+-1}, the rest 0) is produced the same number of times,
namely prod_i (i + 1) / C(n, tau) = tau! times.

Run:  python scripts/17_sampleinball_uniform.py      (seconds)
"""
import itertools, math
from collections import Counter


def sample(n, tau, js, h):
    c = [0] * n
    for k, i in enumerate(range(n - tau, n)):
        j = js[k]
        c[i] = c[j]
        c[j] = -1 if h[k] else 1         # h[i + tau - n] = h[k]
    return tuple(c)


for n, tau in ((6, 2), (8, 3), (10, 4), (12, 5)):
    cnt = Counter()
    ranges = [range(i + 1) for i in range(n - tau, n)]
    for js in itertools.product(*ranges):
        for h in itertools.product((0, 1), repeat=tau):
            cnt[sample(n, tau, js, h)] += 1
    size = math.comb(n, tau) * 2 ** tau
    ok = (len(cnt) == size and len(set(cnt.values())) == 1
          and all(sum(v != 0 for v in c) == tau for c in cnt))
    print(f"n={n:2d} tau={tau}: {sum(cnt.values())} runs, {len(cnt)} outputs (|B_tau| = {size}), "
          f"each {set(cnt.values())} times: {'UNIFORM' if ok else 'NOT UNIFORM'}")
