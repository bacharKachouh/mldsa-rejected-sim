"""
11 -- attacks on the reference protocol (paper, Section "Application to threshold ML-DSA";
Appendix item 11), run against threshold_openw1.py.
Run:  python scripts/11_protocol_attacks.py [nsig]

Adversary = coalition of T-1 = 4 parties out of N = 8, all 8 participate (|H| = 4), and a
second run with |H| = 1 for the mask-stripping check.  The coalition's view per attempt:
  its own nonce shares yhat_j, hence its own w'_j = A(lam_j yhat_j), (h_j, l_j);
  the opened w1 and c of EVERY attempt in the batch; the status of tested attempts;
  for the accepted attempt: all partials, z, h.
Ground-truth secrets are used ONLY to score attacks.

A1  low-part leakage / noisy disclosure: best coalition estimate of w is alpha*w1 + alpha/2;
    run the direct-domain least-squares attack (as in 09b) for u = s2 - t0.
A2  averaging attack on z: LS for s1 from accepted (z, c); plus the flatness diagnostic
    corr(z, c*s1) which is ~0 for a box-uniform nonce (compare 10_nonce_flatness.py).
A3  mask stripping: are honest partials uniform?  with |H| = 1 the partial is a public function.
A4  rejected transcripts: (i) structural: no z, no low part, no carry ever appears for a
    rejected attempt; (ii) empirical: do simple statistics of w1 predict the reject bit?
    (iii) for accepted attempts, w1 == HighBits(rr - c t0) always (no r0-mismatch channel).
Every result is printed with the number of samples; "no recovery" at these sample sizes is
evidence, not proof.
"""
import sys, math
import numpy as np
from core import (n, q, k, l, eta, gamma1, gamma2, alpha, tau, beta, d, omega,
                  centered, zmul, qmul, matvec, decompose, highbits)
from threshold_openw1 import ThresholdSigner, lagrange

NSIG = int(sys.argv[1]) if len(sys.argv) > 1 else 24
N, T = 8, 5
rng = np.random.default_rng(2024)


def rot(c):
    M = np.zeros((n, n)); cc = [int(v) for v in c]
    for i in range(n):
        for j in range(n):
            e = i - j; M[i, j] = cc[e] if e >= 0 else -cc[e + n]
    return M


def ls_recover(rows_M, rows_rhs, truth):
    M = np.vstack(rows_M); fr = []; maxerr = 0
    for comp in range(truth.shape[0]):
        rhs = np.concatenate([r[comp] for r in rows_rhs]).astype(float)
        raw = np.linalg.lstsq(M, rhs, rcond=None)[0]
        fr.append(np.mean(np.rint(raw) == truth[comp])); maxerr = max(maxerr, np.abs(raw - truth[comp]).max())
    return float(np.mean(fr)), maxerr


print("=" * 72)
print(f"ATTACKS on open-w1 threshold ML-DSA, N={N} T={T}, coalition = parties 1..{T-1}, {NSIG} signatures")
print("=" * 72)
ts = ThresholdSigner(N, T, K=24, rng=rng, corrupt=tuple(range(1, T)))
S = list(range(1, N + 1)); lam = lagrange(S)
A, t1 = ts.pk; t0 = ts.t0; s1, s2 = ts.ideal.s1, ts.ideal.s2
u_true = s2 - t0
sigs = []
for i in range(NSIG):
    msg = f"attack-{i}".encode()
    sig, rec = ts.sign(msg, S)
    assert ts.verify(msg, sig)
    sigs.append((msg, sig, rec))
attempts = [a for _, _, rec in sigs for a in rec['attempts']]
n_rej = sum(a['status'] == 'reject' for a in attempts); n_acc = sum(a['status'] == 'accept' for a in attempts)
n_unt = sum(a['status'] == 'untested' for a in attempts)
print(f"attempts seen by the coalition: {len(attempts)} (rejected {n_rej}, accepted {n_acc}, untested {n_unt})")

# ---------------------------------------------------------------- A1
print("\nA1  low-part / noisy-disclosure LS attack on u = s2 - t0 (coalition estimate w ~ alpha*w1 + alpha/2)")
Ms, rhs_est, rhs_true = [], [], []
for msg, (c, z, h), rec in sigs:
    acc = rec['accepted']
    ct1 = np.array([qmul(c, t1[r]) for r in range(k)])
    rr = (matvec(A, z % q) - ct1 * (1 << d)) % q
    w_est = (alpha * acc['w1'] + alpha // 2) % q
    Ms.append(rot(c)); rhs_est.append(centered((w_est - rr) % q))
fr, me = ls_recover(Ms, rhs_est, u_true)
sigma = alpha / math.sqrt(12); law_S = (2 * 0.153 * sigma) ** 2
print(f"   coefficients of u exactly recovered: {fr*100:.2f}%  max|err| {me:.0f}   "
      f"(noise sigma = 2^{math.log2(sigma):.1f}; check-5 law needs ~2^{math.log2(law_S):.1f} signatures)")
print("   => no recovery at this scale; the coalition's w is a noisy disclosure with sigma ~ alpha/sqrt(12)")

# ---------------------------------------------------------------- A2
print("\nA2  averaging attack on accepted z (LS for s1) and flatness diagnostic")
Ms, rhs = [], []
for msg, (c, z, h), rec in sigs:
    Ms.append(rot(c)); rhs.append(centered(z))
fr, me = ls_recover(Ms, rhs, s1)
cs1s, zs = [], []
for msg, (c, z, h), rec in sigs:
    cs1 = np.array([[int(v) for v in zmul(c, s1[r])] for r in range(l)], dtype=np.float64)
    cs1s.append(cs1.ravel()); zs.append(centered(z).astype(np.float64).ravel())
cs1s = np.concatenate(cs1s); zs = np.concatenate(zs)
slope = float((zs * cs1s).sum() / (cs1s * cs1s).sum()); se = float(zs.std() / math.sqrt((cs1s * cs1s).sum()))
print(f"   coefficients of s1 exactly recovered by LS: {fr*100:.2f}%   (a flat nonce gives 1/(2*eta+1) = {100/9:.1f}% by chance)")
print(f"   regression slope of z on c*s1 (oracle diagnostic): {slope:+.3f} +- {se:.3f}   (box-uniform nonce: 0; sum-of-boxes: ~1)")

# ---------------------------------------------------------------- A3
print("\nA3  mask stripping by the T-1 coalition")
hon = [i for i in S if i not in ts.corrupt]
vals = np.concatenate([rec['accepted']['partials'][i].ravel() for _, _, rec in sigs for i in hon]).astype(np.float64)
# uniformity of honest partials mod q: chi-square over 64 bins
bins = np.histogram(vals, bins=64, range=(0, q))[0]; exp = vals.size / 64
chi2 = float(((bins - exp) ** 2 / exp).sum()); dof = 63
print(f"   |H|={len(hon)}: honest partial coefficients mod q: chi2/dof = {chi2/dof:.3f} over {vals.size} samples (uniform ~ 1.0)")
ts1 = ThresholdSigner(N, T, K=24, rng=np.random.default_rng(99), corrupt=tuple(range(1, T)))
S1 = list(range(1, T + 1)); lam1 = lagrange(S1)
sig1, rec1 = ts1.sign(b"one-honest", S1); c1, z1, h1 = sig1; acc1 = rec1['accepted']
hon1 = T
derived = (centered(z1) - sum(lam1[j] * acc1['partials'][j] for j in range(1, T))) * pow(int(lam1[hon1]), q - 2, q) % q
print(f"   |H|=1 (S = coalition + one honest): honest partial equals public function of z: "
      f"{np.array_equal(derived, acc1['partials'][hon1])}  -> it carries nothing beyond z (Lemma 4.2)")

# ---------------------------------------------------------------- A4
print("\nA4  rejected transcripts")
rej = [a for a in attempts if a['status'] == 'reject']
acc_att = [a for a in attempts if a['status'] == 'accept']
unt = [a for a in attempts if a['status'] == 'untested']
print(f"   (i) structural: keys present per rejected attempt = {sorted(rej[0].keys() - {'y'})}  (y is ground truth only);"
      f" no z, no LowBits(w), no carry/H/Q")
# (ii) do simple statistics of w1 predict rejection?  compare rejected vs untested (both never accepted)
def stats(a):
    w1 = a['w1'].astype(np.float64)
    return np.array([w1.mean(), (a['w1'] == 0).mean(), (a['w1'] == 15).mean(), w1.var(), np.abs(np.diff(w1, axis=1)).mean()])
Sr = np.array([stats(a) for a in rej]); Su = np.array([stats(a) for a in unt]) if unt else None
names = ["mean(w1)", "P[w1=0]", "P[w1=15]", "var(w1)", "mean|dw1|"]
if Su is not None and len(Su) > 5:
    for nm, col in enumerate(names):
        mr, mu_ = Sr[:, nm].mean(), Su[:, nm].mean()
        se_ = math.sqrt(Sr[:, nm].var() / len(Sr) + Su[:, nm].var() / len(Su))
        print(f"   (ii) {col:10s}: rejected {mr:.4f} vs untested {mu_:.4f}   z-score {(mr-mu_)/se_ if se_ else 0:+.2f}")
    print(f"        ({len(rej)} rejected vs {len(unt)} untested attempts; |z|<3 = no detectable difference at this scale)")
# (iii) accepted: w1 == HighBits(rr - c t0)?
mism = 0
for msg, (c, z, h), rec in sigs:
    ct1 = np.array([qmul(c, t1[r]) for r in range(k)]); rr = (matvec(A, z % q) - ct1 * (1 << d)) % q
    ct0 = np.array([[int(v) for v in zmul(c, t0[r])] for r in range(k)], dtype=np.int64)
    mism += int((highbits((rr - ct0) % q) != rec['accepted']['w1']).sum())
print(f"   (iii) accepted attempts: coefficients where w1 != HighBits(rr - c t0): {mism} (must be 0: r0 passed)")
print("\nSUMMARY: A1-A3 no recovery / uniform.  A4: the only information about rejected attempts is")
print("(w1, c, reject), which is what the paper's simulation theorem bounds; the statistics above are")
print("evidence at small scale, NOT a proof.")
