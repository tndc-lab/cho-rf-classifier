#!/usr/bin/env python3
"""
cho_stats.py  --  recompute the headline numbers on the FULL dataset, with error bars.
Companion to cho_classifier.py (same folder). Reads events_all.csv.
Run:  cd /scratch/eeduo ; python cho_stats.py
Outputs: stats_results.csv, stats_pairwise.csv, figures_stats/headline_with_errorbars.png
Intervals are 95% ranges from repeating the cross-validation 30 times. They show how
precise the estimate is GIVEN this data; they do NOT capture the larger uncertainty from
having ~1 session per cell line. The honest cross-day result is the real generalization check.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cho_classifier as cc

REPEATS = 30
PURPLE, GRAY = cc.PURPLE, cc.GRAY


def load():
    return pd.read_csv("events_all.csv") if os.path.exists("events_all.csv") else cc.build_table(cc.DATA_ROOT)


def ci(vals):
    a = np.asarray(vals, float)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)


def repeated_cv(df, label, repeats=REPEATS):
    X = cc.Xmat(df); y = df[label].values
    return ci([cc.balanced_acc(*cc.cv_predict(X, y, seed=s)) for s in range(repeats)])


def repeated_role(df, repeats=REPEATS):
    s = df[df.role.isin(["normal", "modified"])]
    if s.role.nunique() < 2:
        return (np.nan, np.nan, np.nan)
    return repeated_cv(s, "role", repeats)


def crossday_combos(df, A="CHOZN Host", B="CHO NIH"):
    HA, HB = df[df.cell_type == A], df[df.cell_type == B]
    da, db = sorted(HA.date.unique()), sorted(HB.date.unique())
    if len(da) < 2 or len(db) < 2:
        return None
    m = lambda d, dt: cc.Xmat(d[d.date == dt])
    out = []
    for ha in da:
        for nb in db:
            teH = [x for x in da if x != ha][0]; teN = [x for x in db if x != nb][0]
            Z = np.vstack([m(HA, ha), m(HB, nb)])
            yy = np.r_[np.zeros((HA.date == ha).sum()), np.ones((HB.date == nb).sum())]
            mu, sd = Z.mean(0), Z.std(0) + 1e-9
            cls, W, b = cc.lda_fit((Z - mu) / sd, yy)
            pH = cc.lda_pred((m(HA, teH) - mu) / sd, cls, W, b)
            pN = cc.lda_pred((m(HB, teN) - mu) / sd, cls, W, b)
            out.append(50 * ((pH == 0).mean() + (pN == 1).mean()))
    return out


def freq_key(f):
    return int("".join(c for c in f if c.isdigit()) or 0)


if __name__ == "__main__":
    ev = load()
    print(f"Headline accuracies with 95% intervals (full data, {REPEATS} CV repeats):\n")
    rows = []
    for f in sorted(ev.freq.unique(), key=freq_key) + ["POOLED"]:
        sub = ev if f == "POOLED" else ev[ev.freq == f]
        m, lo, hi = repeated_cv(sub, "cell_type")
        nm, nlo, nhi = repeated_role(sub)
        combos = crossday_combos(sub)
        rows.append(dict(scope=f, n=len(sub), classes=sub.cell_type.nunique(),
                         cell_type=round(m, 1), ct_lo=round(lo, 1), ct_hi=round(hi, 1),
                         chance=round(100 / sub.cell_type.nunique(), 1),
                         normal_v_modified=None if np.isnan(nm) else round(nm, 1),
                         nvm_lo=None if np.isnan(nm) else round(nlo, 1),
                         nvm_hi=None if np.isnan(nm) else round(nhi, 1),
                         honest_hostnih=None if not combos else round(np.mean(combos), 1),
                         honest_min=None if not combos else round(min(combos), 1),
                         honest_max=None if not combos else round(max(combos), 1)))
        line = f"  {f:7s} cell-type {m:.0f}% [{lo:.0f},{hi:.0f}] (chance {100/sub.cell_type.nunique():.0f}) | nvm "
        line += "n/a" if np.isnan(nm) else f"{nm:.0f}% [{nlo:.0f},{nhi:.0f}]"
        if combos:
            line += f" | honest Host/NIH {np.mean(combos):.0f}% (range {min(combos):.0f}-{max(combos):.0f})"
        print(line)
    pd.DataFrame(rows).to_csv("stats_results.csv", index=False)
    print("\nsaved stats_results.csv")

    print("\nSame-chip pairwise at 11 GHz, with 95% intervals:")
    prow = []
    for a, b in [("CHO PF", "CHO VRC01"), ("CHOZN33", "CHO Trastuzumab"),
                 ("CHO VRC01", "CHOZN23"), ("CHOZN Host", "CHO NIH"), ("CHO-S", "CHO KHL")]:
        s = ev[(ev.freq == "11GHz") & (ev.cell_type.isin([a, b]))]
        if s.cell_type.nunique() < 2:
            print(f"  {a} vs {b}: not both present at 11 GHz"); continue
        m, lo, hi = repeated_cv(s, "cell_type")
        print(f"  {a:11s} vs {b:16s} {m:.0f}% [{lo:.0f},{hi:.0f}]")
        prow.append(dict(pair=f"{a} vs {b}", acc=round(m, 1), lo=round(lo, 1), hi=round(hi, 1)))
    pd.DataFrame(prow).to_csv("stats_pairwise.csv", index=False)
    print("saved stats_pairwise.csv")

    e11 = ev[ev.freq == "11GHz"]
    items = []
    m, lo, hi = repeated_cv(e11, "cell_type")
    items.append((f"guess cell type\n({e11.cell_type.nunique()} types)", m, m - lo, hi - m, 100 / e11.cell_type.nunique()))
    nm, nlo, nhi = repeated_role(e11)
    if not np.isnan(nm):
        items.append(("normal vs\nmodified", nm, nm - nlo, nhi - nm, 50))
    combos = crossday_combos(e11)
    if combos:
        hh = np.mean(combos)
        items.append(("fair test:\nnew days", hh, hh - min(combos), max(combos) - hh, 50))
    labels = [i[0] for i in items]; vals = [i[1] for i in items]
    yerr = [[i[2] for i in items], [i[3] for i in items]]; chances = [i[4] for i in items]
    os.makedirs("figures_stats", exist_ok=True)
    x = np.arange(len(items))
    plt.figure(figsize=(7.5, 4.4))
    plt.bar(x, vals, color=PURPLE, yerr=yerr, capsize=7, ecolor="#333")
    plt.plot(x, chances, "k_", ms=24, label="chance")
    for i, v in enumerate(vals):
        plt.text(i, v + yerr[1][i] + 2, f"{v:.0f}%", ha="center", fontsize=10)
    plt.xticks(x, labels); plt.ylim(0, 100); plt.ylabel("balanced accuracy %")
    plt.title("11 GHz headline results, with 95% intervals"); plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig("figures_stats/headline_with_errorbars.png", dpi=140); plt.close()
    print("\nsaved figures_stats/headline_with_errorbars.png")