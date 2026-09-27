"""
09b -- empirical law of the least-squares attack on noisy disclosures of the low part
(paper: remark after Lemma "the carry must stay private"; Appendix item 9).  With noise sigma
per coefficient the recovery error behaves as C sigma / sqrt(S) after S signatures; the fitted
C (about 0.153) gives S ~ (2 C sigma)^2 = (0.31 sigma)^2 for exact recovery.
Run:  python scripts/09b_noisy_disclosure_law.py
"""
import numpy as np, math
from core import *
from mldsa_ref import keygen
rng=np.random.default_rng(42)
pk,sk=keygen(rng); A,s1,s2,t0,t1=sk
utrue=centered(s2-t0)
def rotmat(cp):
    M=np.zeros((n,n)); cc=[int(v) for v in cp]
    for j in range(n):
        for i_,v in enumerate(cc):
            if v:
                p=i_+j; M[p if p<n else p-n, j]+= v if p<n else -v
    return M
def sign_lp(msg):
    while True:
        y=rng.integers(-gamma1+1,gamma1+1,size=(l,n),dtype=np.int64)
        w=matvec(A,y%q); c=challenge(msg,highbits(w))
        z=y+np.array([[int(v) for v in zmul(c,s1[j])] for j in range(l)],dtype=np.int64)
        if np.abs(centered(z)).max()>=gamma1-beta: continue
        cs2=np.array([[int(v) for v in zmul(c,s2[i])] for i in range(k)],dtype=np.int64)
        if np.abs(centered(decompose(w)[1]-cs2)).max()>=gamma2-beta: continue
        return c,z%q,w
NP=200
POOL=[sign_lp(f"big{i}".encode()) for i in range(NP)]
pre=[]
for c,z,w in POOL:
    ct1=np.array([qmul(c,t1[i]) for i in range(k)]); r=(matvec(A,z)-ct1*(1<<d))%q
    pre.append((centered(w),centered(r),rotmat(c)))
sg=int(2**11.5)   # max noise the hint budget allows
print(f"noise sigma = 2^11.5 = {sg} (max absorbable by hint slack). ||u||inf = {np.abs(utrue).max()}")
print("LS recovery error vs number of signatures S (component 0):")
for S in (20,50,100,200):
    Ms=np.vstack([pre[s][2] for s in range(S)])
    i=0; rhs=[]
    for s in range(S):
        wc,rc,_=pre[s]
        noise=np.rint(rng.normal(0,sg,n)).astype(np.int64)
        rhs.append(centered((wc[i]+noise-rc[i])%q))
    rhs=np.concatenate(rhs).astype(float)
    uh=np.rint(np.linalg.lstsq(Ms,rhs,rcond=None)[0]).astype(np.int64)
    err=np.abs(uh-utrue[i])
    print(f"  S={S:3d}: max|err|={err.max():4d}  mean|err|={err.mean():5.2f}  exact coefs={np.mean(uh==utrue[i])*100:5.1f}%")
# extrapolate: error ~ sigma/sqrt(S) * const. When does max|err| < 0.5 (=> exact)?
print("\nerror ~ C*sigma/sqrt(S); solve for S giving exact recovery")
S=200; Ms=np.vstack([pre[s][2] for s in range(S)])
i=0; rhs=[]
for s in range(S):
    wc,rc,_=pre[s]; noise=np.rint(rng.normal(0,sg,n)).astype(np.int64)
    rhs.append(centered((wc[i]+noise-rc[i])%q))
uh_raw=np.linalg.lstsq(Ms,np.concatenate(rhs).astype(float),rcond=None)[0]
resid_std=np.std(uh_raw-utrue[i])
C=resid_std*math.sqrt(S)/sg
S_break=(C*sg/0.4)**2
print(f"  fitted C={C:.3f}, extrapolated S for exact recovery ~ {S_break:.0f} signatures")
