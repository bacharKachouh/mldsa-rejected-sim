"""Faithful-enough ML-DSA-65 signer to measure the REAL hint-weight distribution."""
import numpy as np, hashlib
from core import (n,q,k,l,eta,gamma1,gamma2,alpha,tau,beta,d,omega,
                  centered,zmul,qmul,matvec,mono,power2round,decompose,highbits,usehint,challenge)

def makehint(z0, r):
    # FIPS 204 MakeHint: does adding z0 change the high bits of r?
    r1 = highbits(r)
    v1 = highbits((r + z0) % q)
    return (r1 != v1).astype(np.int64)

def keygen(rng):
    A = rng.integers(0, q, size=(k, l, n), dtype=np.int64)
    s1 = rng.integers(-eta, eta+1, size=(l, n), dtype=np.int64)
    s2 = rng.integers(-eta, eta+1, size=(k, n), dtype=np.int64)
    t = (matvec(A, s1 % q) + s2) % q
    t1, t0 = power2round(t)
    return (A, t1), (A, s1, s2, t0, t1)

def sign(sk, msg, rng, cap=1000):
    A, s1, s2, t0, t1 = sk
    for att in range(1, cap+1):
        y = rng.integers(-gamma1+1, gamma1+1, size=(l, n), dtype=np.int64)
        w = matvec(A, y % q)
        w1 = highbits(w)
        c = challenge(msg, w1)
        cs1 = np.array([[int(v) for v in zmul(c, s1[j])] for j in range(l)], dtype=np.int64)
        z = y + cs1
        if np.abs(centered(z)).max() >= gamma1 - beta:      # z-bound
            continue
        cs2 = np.array([[int(v) for v in zmul(c, s2[i])] for i in range(k)], dtype=np.int64)
        r0 = decompose(w)[1]
        if np.abs(centered((r0 - cs2))).max() >= gamma2 - beta:   # r0 check
            continue
        ct0 = np.array([[int(v) for v in zmul(c, t0[i])] for i in range(k)], dtype=np.int64)
        h = makehint((-cs2 + ct0) % q, (w - cs2) % q)
        if h.sum() > omega:
            continue
        return (c, z % q, h), att, h.sum()
    return None, cap, None

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    pk, sk = keygen(rng)
    atts, hw, ok = [], [], 0
    for i in range(60):
        sig, a, wt = sign(sk, f"m{i}".encode(), rng)
        # verify
        A, t1 = pk; c, z, h = sig
        ct1 = np.array([qmul(c, t1[i2]) for i2 in range(k)])
        rr = (matvec(A, z % q) - ct1*(1<<d)) % q
        v = all(int(x)==int(yv) for x,yv in zip(challenge(f"m{i}".encode(), usehint(h, rr)), c))
        ok += v; atts.append(a); hw.append(wt)
    hw = np.array(hw)
    print(f"valid {ok}/60   mean attempts {np.mean(atts):.2f}")
    print(f"hint weight: mean {hw.mean():.1f}  max {hw.max()}  omega {omega}")
    print(f"SLACK omega - E[wt] = {omega - hw.mean():.1f}  coefficients of free correction budget")
    print(f"per-coef hint density p = E[wt]/nk = {hw.mean()/(n*k):.5f}")
