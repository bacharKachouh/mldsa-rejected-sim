"""
threshold_openw1.py -- the "open-w1 / public-hash" T-of-N threshold ML-DSA-65 signer described in
the paper's Section "Application to threshold ML-DSA".  Functional (data-flow exact)
implementation; it is not a secure computation and has no security proof.

WHAT IS REAL HERE
  * key sharing: uniform Shamir over R_q at points x_i = i in Z_q (perfect privacy, any T-1 shares)
  * nonce sharing: additive-mod-q shares of a box-uniform y, Lagrange-prescaled
  * w1 via the exact carry identity from the parties' floor decompositions (h_i, l_i)
    -- checked against core.highbits on every attempt (assert)
  * open-w1 -> public hash -> parallel batch of K attempts -> open ONE attempt's partials
  * hint and verification are stock FIPS-204 (core.verify), t0 public
  * a per-signature ROUND COUNTER and a full TRANSCRIPT (what a T-1 coalition sees)

WHAT IS A STUB (marked F_*): the secure computations are done in the clear on reconstructed
values by the class Ideal below.  Nothing in this file is an MPC; the MP-SPDZ programs that
realise the stubbed steps are in bench/.
  F_nonce : samples y and its additive shares                   (offline, message-independent)
  F_w1    : evaluates the carry identity on (h_i, l_i) sums       (offline, message-independent)
  F_test  : z-test, r0-test and hint-weight test on shares, returns the FIRST accepting
            attempt index; statuses of earlier attempts become public, which is the
            rejected-attempt view the paper bounds                   (online, D_TEST rounds)
  F_dkg   : samples the key with eta-bounded coefficients in shared form  (one-time)
"""
import numpy as np
from core import (n, q, k, l, eta, gamma1, gamma2, alpha, tau, beta, d, omega,
                  centered, zmul, qmul, matvec, power2round, decompose, highbits, challenge, verify)
from mldsa_ref import makehint

D_TEST_GMW = 10   # AND-depth of the batched comparison circuit (counted in bench/counts.py)
D_TEST_GC = 2     # with self-generated function-dependent (garbled) preprocessing


# --------------------------------------------------------------------------- sharing
def lagrange(S):
    """lambda_i for reconstruction at 0 from points x_i = i (i in S), over Z_q."""
    lam = {}
    for i in S:
        num = den = 1
        for j in S:
            if j != i:
                num = num * j % q
                den = den * ((j - i) % q) % q
        lam[i] = num * pow(den, q - 2, q) % q
    return lam


def shamir_share(v, N, T, rng):
    """v: int array (rows, n) mod q -> dict i -> share (rows, n) int64 mod q, degree T-1."""
    v = np.asarray(v, dtype=np.int64) % q
    coefs = [v] + [rng.integers(0, q, size=v.shape, dtype=np.int64) for _ in range(T - 1)]
    shares = {}
    for i in range(1, N + 1):
        acc = np.zeros(v.shape, dtype=np.int64)
        for j, cj in enumerate(coefs):
            acc = (acc + cj * pow(i, j, q)) % q       # cj < 2^23, pow < 2^23: fits int64
        shares[i] = acc
    return shares


def reconstruct(shares, S, lam):
    acc = None
    for i in S:
        term = shares[i] * lam[i] % q
        acc = term if acc is None else (acc + term) % q
    return acc


# --------------------------------------------------------------------------- ideal stubs
class Ideal:
    """The trusted computations.  Holds the key (as F_att does); never leaks beyond outputs."""

    def __init__(self, rng, N, T):
        self.rng, self.N, self.T = rng, N, T

    # F_dkg (stub: centrally sampled; a real DKG samples the same distribution in shared form)
    def dkg(self):
        rng = self.rng
        A = rng.integers(0, q, size=(k, l, n), dtype=np.int64)
        s1 = rng.integers(-eta, eta + 1, size=(l, n), dtype=np.int64)
        s2 = rng.integers(-eta, eta + 1, size=(k, n), dtype=np.int64)
        t = (matvec(A, s1 % q) + s2) % q
        t1, t0 = power2round(t)
        self.A, self.s1, self.s2, self.t0, self.t1 = A, s1, s2, t0, t1
        sh1 = shamir_share(s1, self.N, self.T, rng)
        sh2 = shamir_share(s2, self.N, self.T, rng)
        return (A, t1), t0, {i: (sh1[i], sh2[i]) for i in sh1}

    # F_nonce: y uniform in the FIPS box, additive-mod-q shares u_i, prescaled yhat_i = lam_i^-1 u_i
    def nonce(self, S, lam):
        rng = self.rng
        y = rng.integers(-gamma1 + 1, gamma1 + 1, size=(l, n), dtype=np.int64)
        u = {i: rng.integers(0, q, size=(l, n), dtype=np.int64) for i in S[:-1]}
        u[S[-1]] = (y - sum(u.values())) % q
        yhat = {i: u[i] * pow(int(lam[i]), q - 2, q) % q for i in S}
        return y, u, yhat

    # F_w1: carry identity on the parties' clean decompositions; returns additive shares of w1
    def w1_from_shares(self, wprime, S):
        H = np.zeros((k, n), dtype=object); L = np.zeros((k, n), dtype=object); WZ = np.zeros((k, n), dtype=object)
        for i in S:
            wi = wprime[i] % q
            H += (wi // alpha).astype(object); L += (wi % alpha).astype(object); WZ += wi.astype(object)
        Q = WZ // q
        m = (L - Q) // alpha                                   # Lemma 1: m in {-1..T-1}, key-independent
        hclean = H - 16 * Q + m                                # clean HighBits of w = WZ mod q
        w = (WZ % q).astype(np.int64)
        lclean = w % alpha
        assert np.array_equal(hclean.astype(np.int64), w // alpha), "carry identity violated"
        # floor decomposition -> FIPS centred high part
        up = lclean > alpha // 2
        w1 = (hclean.astype(np.int64) + up.astype(np.int64)) % 16
        assert np.array_equal(w1, highbits(w)), "clean->FIPS map violated"
        # hand out additive shares of w1 (opened online in round 1)
        sh = {i: self.rng.integers(0, q, size=(k, n), dtype=np.int64) for i in S[:-1]}
        sh[S[-1]] = (w1 - sum(sh.values())) % q
        return w1, w, sh

    # F_test: FIPS acceptance for each attempt in the batch; returns first accepting index
    def first_accepting(self, msg, attempts, cs):
        statuses = []
        for j, (att, c) in enumerate(zip(attempts, cs)):
            y, w = att['y'], att['w']
            cs1 = np.array([[int(v) for v in zmul(c, self.s1[r])] for r in range(l)], dtype=np.int64)
            z = y + cs1
            if np.abs(centered(z)).max() >= gamma1 - beta:
                statuses.append('reject'); continue
            cs2 = np.array([[int(v) for v in zmul(c, self.s2[r])] for r in range(k)], dtype=np.int64)
            if np.abs(centered(decompose(w)[1] - cs2)).max() >= gamma2 - beta:
                statuses.append('reject'); continue
            ct0 = np.array([[int(v) for v in zmul(c, self.t0[r])] for r in range(k)], dtype=np.int64)
            h = makehint((-cs2 + ct0) % q, (w - cs2) % q)
            if h.sum() > omega:
                statuses.append('reject'); continue
            statuses.append('accept')
            return j, statuses
        return None, statuses


# --------------------------------------------------------------------------- protocol
class ThresholdSigner:
    def __init__(self, N, T, K=24, rng=None, corrupt=()):
        self.N, self.T, self.K = N, T, K
        self.rng = rng or np.random.default_rng(0)
        self.ideal = Ideal(self.rng, N, T)
        self.pk, self.t0, self.key_shares = self.ideal.dkg()
        self.A, self.t1 = self.pk
        self.corrupt = tuple(corrupt)                       # indices the adversary controls (view)
        self.transcript = []                                 # per signature

    # ---- offline: message-independent batch of K attempts for signing set S
    def offline(self, S):
        lam = lagrange(S)
        batch = []
        for _ in range(self.K):
            y, u, yhat = self.ideal.nonce(S, lam)
            wprime = {i: matvec(self.A, u[i]) for i in S}    # each party: A * (lam_i * yhat_i) locally
            w1, w, w1sh = self.ideal.w1_from_shares(wprime, S)
            batch.append(dict(y=y, u=u, yhat=yhat, w=w, w1=w1, w1sh=w1sh))
        return lam, batch

    # ---- online: rounds counted
    def sign(self, msg, S, test_depth=D_TEST_GMW):
        rounds = 0
        record = dict(msg=msg, S=S, attempts=[], rounds=0, batches=0)
        while True:
            lam, batch = self.offline(S)
            record['batches'] += 1
            # ROUND 1: open w1 of every attempt in the batch, hash publicly
            rounds += 1
            w1s = [sum(att['w1sh'][i] for i in S) % q for att in batch]
            for att, w1o in zip(batch, w1s):
                assert np.array_equal(w1o, att['w1'])
            cs = [challenge(msg, w1o) for w1o in w1s]
            # ROUNDS 2..1+D: F_test on shares
            rounds += test_depth
            jstar, statuses = self.ideal.first_accepting(msg, batch, cs)
            for j, (att, c) in enumerate(zip(batch, cs)):
                st = statuses[j] if j < len(statuses) else 'untested'
                record['attempts'].append(dict(w1=att['w1'], c=c, status=st, y=att['y'],
                                               corrupt_yhat={i: att['yhat'][i] for i in self.corrupt if i in S}))
            if jstar is None:
                continue                                    # whole batch failed (prob ~0.8^K): new batch
            att, c = batch[jstar], cs[jstar]
            # ROUND 2+D: broadcast partials of the accepted attempt
            rounds += 1
            partials = {i: (att['yhat'][i] + np.array([qmul(c, self.key_shares[i][0][r]) for r in range(l)])) % q
                        for i in S}
            z = centered(reconstruct(partials, S, lam))
            assert np.array_equal(z, att['y'] + np.array([[int(v) for v in zmul(c, self.ideal.s1[r])] for r in range(l)]))
            ct1 = np.array([qmul(c, self.t1[r]) for r in range(k)])
            rr = (matvec(self.A, z % q) - ct1 * (1 << d)) % q
            ct0 = np.array([[int(v) for v in zmul(c, self.t0[r])] for r in range(k)], dtype=np.int64)
            h = makehint((-ct0) % q, rr)                     # public: t0 is public in this model
            sig = (c, z % q, h)
            record['rounds'] = rounds
            record['accepted'] = dict(index=jstar, z=z, h=h, partials=partials, w1=att['w1'], c=c)
            self.transcript.append(record)
            return sig, record

    def verify(self, msg, sig):
        return verify(self.pk, msg, sig)


if __name__ == "__main__":
    import sys, time
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    T = int(sys.argv[2]) if len(sys.argv) > 2 else N // 2 + 1
    nsig = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ts = ThresholdSigner(N, T, K=24, rng=np.random.default_rng(N), corrupt=tuple(range(1, T)))
    S = list(range(1, N + 1))
    ok = 0; rounds = []; rej = []
    t = time.time()
    for i in range(nsig):
        sig, rec = ts.sign(f"msg{i}".encode(), S)
        ok += ts.verify(f"msg{i}".encode(), sig); rounds.append(rec['rounds'])
        rej.append(sum(a['status'] == 'reject' for a in rec['attempts']))
    print(f"N={N} T={T} |S|={len(S)}: {ok}/{nsig} signatures verify under stock FIPS-204 verifier; "
          f"online rounds/signature (GMW test depth) = {rounds}; rejected statuses revealed = {rej}; "
          f"{time.time()-t:.1f}s")
