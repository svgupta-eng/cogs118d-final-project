"""Preliminary MLE fits for Palminteri et al. (2017) replication.

Complete feedback (P2017b): Full model (aC+, aC-, aU+, aU-, beta) and
Confirmation model (aCON, aDIS, beta). Partial feedback (P2017a): (aC+, aC-, beta).
Q starts at 0, outcomes are +1/-1, Q-values are separate for each of the 8 cue pairs.
Plain MLE with bounds a in [0,1], beta in [0,100], 20 random starts per participant.
"""
import sys
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from scipy import stats

DATA, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)

def load(f):
    m = sio.loadmat(f"{DATA}/{f}")
    keys = [k for k in ["cho", "out", "cou", "sta"] if k in m]
    return [{k: m[k][0, s].ravel().astype(int) for k in keys} for s in range(m["cho"].shape[1])]

def nll(params, d, model):
    if model == "partial":
        acp, acm, beta = params; aup = aum = None
    elif model == "full":
        acp, acm, aup, aum, beta = params
    elif model == "confirm":
        acon, adis, beta = params
        acp, acm, aup, aum = acon, adis, adis, acon  # confirmatory: chosen+ and unchosen-
    Q = np.zeros((9, 3))
    ll = 0.0
    cou = d.get("cou")
    for t in range(len(d["cho"])):
        s, c = d["sta"][t], d["cho"][t]
        u = 3 - c
        dq = Q[s, c] - Q[s, u]
        ll += -np.logaddexp(0, -beta * dq)  # log softmax for the chosen option
        pe = d["out"][t] - Q[s, c]
        Q[s, c] += (acp if pe > 0 else acm) * pe
        if model != "partial":
            peu = cou[t] - Q[s, u]
            Q[s, u] += (aup if peu > 0 else aum) * peu
    return -ll

def fit(d, model, k):
    bounds = [(0, 1)] * (k - 1) + [(0, 100)]
    best = None
    for _ in range(20):
        x0 = np.r_[rng.uniform(0, 1, k - 1), rng.uniform(0.5, 15)]
        r = minimize(nll, x0, args=(d, model), bounds=bounds, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    n = len(d["cho"])
    return best.x, best.fun, 2 * best.fun + k * np.log(n)

lines = []
part = load("P2017a.mat")
P = np.array([fit(d, "partial", 3)[0] for d in part])
t, p = stats.ttest_rel(P[:, 0], P[:, 1])
lines += ["## Partial feedback (N=20): aC+ vs aC-",
          f"aC+ = {P[:,0].mean():.3f} ± {P[:,0].std(ddof=1)/np.sqrt(20):.3f}, aC- = {P[:,1].mean():.3f} ± {P[:,1].std(ddof=1)/np.sqrt(20):.3f}",
          f"paired t(19) = {t:.2f}, p = {p:.2g}"]

comp = load("P2017b.mat")
F, bicF, bicC, C = [], [], [], []
for d in comp:
    x, _, b = fit(d, "full", 5); F.append(x); bicF.append(b)
    x, _, b = fit(d, "confirm", 3); C.append(x); bicC.append(b)
F, C, bicF, bicC = map(np.array, (F, C, bicF, bicC))
sem = lambda v: v.std(ddof=1) / np.sqrt(len(v))
lines += ["## Complete feedback (N=20): Full model"]
for i, nm in enumerate(["aC+", "aC-", "aU+", "aU-", "beta"]):
    lines.append(f"{nm} = {F[:,i].mean():.3f} ± {sem(F[:,i]):.3f}")
t1, p1 = stats.ttest_rel(F[:, 0], F[:, 1]); t2, p2 = stats.ttest_rel(F[:, 3], F[:, 2])
lines += [f"aC+ vs aC-: t(19) = {t1:.2f}, p = {p1:.2g}", f"aU- vs aU+: t(19) = {t2:.2f}, p = {p2:.2g}"]
t3, p3 = stats.ttest_rel(C[:, 0], C[:, 1])
lines += ["## Complete feedback: Confirmation model",
          f"aCON = {C[:,0].mean():.3f} ± {sem(C[:,0]):.3f}, aDIS = {C[:,1].mean():.3f} ± {sem(C[:,1]):.3f}",
          f"aCON vs aDIS: t(19) = {t3:.2f}, p = {p3:.2g}",
          f"Summed BIC: Full = {bicF.sum():.1f}, Confirmation = {bicC.sum():.1f}, dBIC (Full - Conf) = {bicF.sum()-bicC.sum():.1f}",
          f"Participants best fit: Confirmation {np.sum(bicC < bicF)}/20",
          f"beta at upper bound (>99): Full {np.sum(F[:,4] > 99)}, Conf {np.sum(C[:,2] > 99)}"]
print("\n".join(lines))
open(f"{OUT}/prelim_fit_stats.txt", "w").write("\n".join(lines) + "\n")
