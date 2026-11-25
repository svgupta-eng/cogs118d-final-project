"""Behavioral signature of Palminteri et al. (2017) from the open data (hrl-team/decay_1).

Data coding, inferred from the outcomes themselves:
  con 1 = Symmetric 50/50, con 2 and 3 = Asymmetric 25/75 (option 2 better),
  con 4 = Reversal 17/83 for trials 1-12 of a pair, then 83/17.
  sta 1-8 = the 8 cue pairs, 4 contexts x 2 sessions, 24 trials each.
P2017a = partial feedback (chosen outcome only), P2017b = complete feedback.
"""
import sys
import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1]
OUT = sys.argv[2]

def load(f):
    m = sio.loadmat(f"{DATA}/{f}")
    subs = []
    for s in range(m["cho"].shape[1]):
        subs.append({k: m[k][0, s].ravel().astype(int) for k in ["cho", "con", "sta"]})
    return subs

def measures(sub):
    cho, con, sta = sub["cho"], sub["con"], sub["sta"]
    pref, asym, rev1, rev2 = [], [], [], []
    curve = np.zeros(24)
    nrev = 0
    for st in np.unique(sta):
        idx = np.where(sta == st)[0]          # trials of this pair, in order
        c = con[idx[0]]
        ch = cho[idx]
        if c == 1:   # preferred choice rate: share of the more-chosen option
            p1 = np.mean(ch == 1)
            pref.append(max(p1, 1 - p1))
        elif c in (2, 3):
            asym.append(np.mean(ch == 2))
        elif c == 4:
            correct = np.where(np.arange(24) < 12, ch == 2, ch == 1)
            rev1.append(correct[:12].mean())
            rev2.append(correct[12:].mean())
            curve += correct
            nrev += 1
    return dict(pref=np.mean(pref), asym=np.mean(asym), rev1=np.mean(rev1),
                rev2=np.mean(rev2), curve=curve / nrev)

groups = {"Partial feedback": load("P2017a.mat"), "Complete feedback": load("P2017b.mat")}
res = {g: [measures(s) for s in subs] for g, subs in groups.items()}
keys = ["pref", "asym", "rev1", "rev2"]
labels = ["Symmetric 50/50\npreferred choice", "Asymmetric 75/25\ncorrect choice",
          "Reversal\n1st half correct", "Reversal\n2nd half correct"]

lines = []
for g in res:
    arr = {k: np.array([r[k] for r in res[g]]) for k in keys}
    lines.append(f"## {g} (N={len(res[g])})")
    for k in keys:
        x = arr[k]
        sem = x.std(ddof=1) / np.sqrt(len(x))
        t, p = stats.ttest_1samp(x, 0.5)
        lines.append(f"{k}: mean={x.mean():.3f} SEM={sem:.3f}  vs 0.5: t({len(x)-1})={t:.2f} p={p:.2g}")
    t, p = stats.ttest_rel(arr["rev1"], arr["rev2"])
    lines.append(f"reversal 1st vs 2nd half (paired): t({len(x)-1})={t:.2f} p={p:.2g}")
lines.append("## Partial vs complete (Welch, independent groups)")
for k in keys:
    a = np.array([r[k] for r in res["Partial feedback"]])
    b = np.array([r[k] for r in res["Complete feedback"]])
    t, p = stats.ttest_ind(a, b, equal_var=False)
    lines.append(f"{k}: partial={a.mean():.3f} complete={b.mean():.3f}  t={t:.2f} p={p:.2g}")
print("\n".join(lines))
open(f"{OUT}/signature_stats.txt", "w").write("\n".join(lines) + "\n")

# ---- figure ----
COL = {"Partial feedback": "#2a78d6", "Complete feedback": "#eb6834"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
fig.patch.set_facecolor(SURF)
rng = np.random.default_rng(0)
w = 0.36
for gi, g in enumerate(res):
    for ki, k in enumerate(keys):
        x = np.array([r[k] for r in res[g]])
        pos = ki + (gi - 0.5) * (w + 0.04)
        ax1.bar(pos, x.mean(), w, color=COL[g], alpha=0.85, label=g if ki == 0 else None, zorder=2)
        ax1.errorbar(pos, x.mean(), yerr=x.std(ddof=1) / np.sqrt(len(x)), color=INK, capsize=4, lw=1.5, zorder=4)
        ax1.scatter(pos + rng.uniform(-0.1, 0.1, len(x)), x, s=14, color=INK, alpha=0.35, zorder=3, linewidths=0)
ax1.axhline(0.5, color=INK2, ls="--", lw=1, zorder=1)
ax1.plot([-0.42, 0.42], [0.58, 0.58], color=INK, lw=1.5, ls="-.", zorder=5, label="Random-chooser baseline (0.58)")
ax1.text(3.75, 0.51, "chance", color=INK2, fontsize=11, ha="right", va="bottom")
ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylim(0, 1.05); ax1.set_ylabel("Choice rate (proportion of trials)")
ax1.set_title("Choice rates by context (mean ± SEM, dots = participants)", fontsize=13, color=INK)
ax1.legend(frameon=False, loc="lower left", fontsize=11)

t = np.arange(1, 25)
for g in res:
    c = np.array([r["curve"] for r in res[g]])
    m, s = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
    ax2.plot(t, m, color=COL[g], lw=2, label=g)
    ax2.fill_between(t, m - s, m + s, color=COL[g], alpha=0.18, linewidth=0)
ax2.axvline(12.5, color=INK2, ls=":", lw=1.2)
ax2.text(12.8, 0.97, "contingency\nreverses", color=INK2, fontsize=10, va="top")
ax2.axhline(0.5, color=INK2, ls="--", lw=1)
ax2.set_xlabel("Trial within reversal pair"); ax2.set_ylabel("P(choose currently better option)")
ax2.set_ylim(0, 1.05); ax2.set_xlim(1, 24)
ax2.set_title("Reversal context learning curve", fontsize=13, color=INK)
ax2.legend(frameon=False, loc="lower left", fontsize=11)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/behavioral_signature.png", dpi=200, facecolor=SURF)
